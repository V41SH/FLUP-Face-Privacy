import os 
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import re
from cropper import *
import torch.nn.functional as F


class CelebADataset(Dataset):
    def __init__(self, transform=None, faceTransform=None, dims=128, faceFactor=0.7, basicCrop=False):
        self.basicCrop = basicCrop
        self.faceFactor = faceFactor
        self.dims = dims
        self.img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../celebA/Img/img_celeba/"))

        self.transform = transform
        self.image_filenames = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]
        self.faceTransform = faceTransform

        with open(os.path.join(self.img_dir, "../../Anno/identity_CelebA.txt"), "r") as file:
            lines = file.readlines()
            self.identity = {line.split()[0]: int(line.split()[1]) for line in lines[2:]}

        with open(os.path.join(self.img_dir, "../../Anno/list_bbox_celeba.txt"), "r") as file:
            lines = file.readlines()
            self.bbox_anno = {line.split()[0]: list(map(int, line.split()[1:])) for line in lines[2:]}

        if not self.basicCrop:
            with open(os.path.join(os.path.dirname(__file__), "landmark.txt"), "r") as file:
                line = file.readline()
                n_landmark = len(re.split('[ ]+', line)[1:]) // 2

            self.landmark_anno = np.genfromtxt(os.path.join(os.path.dirname(__file__), "landmark.txt"), dtype=float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1, n_landmark, 2)
            self.std_landmark_anno = np.genfromtxt(os.path.join(os.path.dirname(__file__), "standard_landmark_68pts.txt"), dtype=float).reshape(n_landmark, 2)


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image)

        # if self.faceTransform:

        if self.basicCrop:
            x, y, w, h = self.bbox_anno[filename]

            # scale the image a little - ensure that the face is approximately the same size, 
            # according to the final crop size. assuming square crop size
            scale = self.dims / (h * 1.3) # to scale the image to approximately the same size as crop
            image = np.array(image.permute(1, 2, 0)*255, dtype=np.uint8)
            image = cv2.resize(image, [int(image.shape[1] * scale), int(image.shape[0] * scale)])
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # Normalize back to [0,1]

            # Get face bounding box
            x = int(x * scale) 
            y = int(y * scale) 
            w = int(w * scale) 
            h = int(h * scale) 

            # want to shrink top left corner more towards the top left
            xp = max(0, int(x - (w * (1 - self.faceFactor) / 2)))
            yp = max(0, int(y - (h * (1 - self.faceFactor) / 2)))

            # Increase the width and height
            wp = int(w * (2 - self.faceFactor)) 
            hp = int(h * (2 - self.faceFactor)) 

            if xp + wp > image.shape[2] or yp + hp > image.shape[1]:  # image.shape[2] is width, image.shape[1] is height
                padding = max(xp + wp - image.shape[2], yp + hp - image.shape[1]) + 1
                image = F.pad(image, (padding, padding, padding, padding), mode='constant', value=0)
            else:
                padding = 0

            if self.faceTransform:
                image[:, y + padding:y + padding + h, x + padding:x + padding + w] = self.faceTransform(image[:, y + padding:y + padding + h, x + padding:x + padding + w])
            image = image[:, yp + padding : yp + padding + hp, xp + padding : xp + padding + wp]

            image = np.array(image.permute(1, 2, 0), dtype=np.uint8)
            # print(image.shape)
            if image.shape[0] == 0: 
                print(wp, hp, yp + padding, yp + padding + hp)
                # cv2.imshow("bruh", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            image = cv2.resize(image, [self.dims, self.dims])
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # Normalize back to [0,1]


        else:
            # Get face bounding box
            x, y, w, h = self.bbox_anno[filename]

            # scale the image a little - ensure that the face is approximately the same size, 
            # according to the final crop size. assuming square crop size
            scale = self.dims / (h * 1.3) # to scale the image to approximately the same size as crop
            image = np.array(image.permute(1, 2, 0)*255, dtype=np.uint8)
            image = cv2.resize(image, [int(image.shape[1] * scale), int(image.shape[0] * scale)])

            # Get face bounding box
            x = int(x * scale) 
            y = int(y * scale) 
            w = int(w * scale) 
            h = int(h * scale) 

            image = torch.from_numpy(image).permute(2, 0, 1).float()  # Normalize back to [0,1]

            # put the blurred image back 
            if self.faceTransform:
                image[:, y:y+h, x:x+w] = self.faceTransform(image[:, y:y+h, x:x+w])

            # do the landmark stuff
            image = align_crop_opencv(np.array(image.permute(1, 2, 0), dtype=np.uint8),  # Convert tensor to OpenCV format
                                            self.landmark_anno[int(filename[:-4])-1] * scale,  # Get source landmarks
                                            self.std_landmark_anno,
                                            crop_size=self.dims,
                                            face_factor=self.faceFactor)

            # convert back to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # Normalize back to [0,1]

        return image, self.identity[filename]


class CelebADual(): 
    def __init__(self, transform=None, faceTransform=None, dims=128, faceFactor=0.7, basicCrop=False, batch_size=32, shuffle=True):
        self.unBlurDataset = CelebADataset(transform=transform, faceTransform=faceTransform, dims=dims, faceFactor=faceFactor, basicCrop=basicCrop)
        self.BlurDataset = CelebADataset(transform=transform, faceTransform=None, dims=dims, faceFactor=faceFactor, basicCrop=basicCrop)

        # Create dataloaders for both versions
        self.unBlurLoader = DataLoader(self.unBlurDataset, batch_size=batch_size, shuffle=shuffle)
        self.BlurLoader = DataLoader(self.BlurDataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        # Zip the two dataloaders so they return corresponding batches
        return zip(iter(self.unBlurLoader), iter(self.BlurLoader))

    
