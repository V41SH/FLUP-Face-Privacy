import os, sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import re
from cropper import *
import random
# from celebA_dataloader.cropper import *
import torch.nn.functional as F

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from utils.blurring_utils import blur_face


class CelebADataset(Dataset):
    def __init__(self, transform=None, faceTransform=None, dims=128, faceFactor=0.7, crop="neural", triplet=False):
        self.crop = crop 
        self.faceFactor = faceFactor
        self.dims = dims
        self.triplet = triplet
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

        if self.crop == 'transform':
            with open(os.path.join(os.path.dirname(__file__), "landmark.txt"), "r") as file:
                line = file.readline()
                n_landmark = len(re.split('[ ]+', line)[1:]) // 2

            self.landmark_anno = np.genfromtxt(os.path.join(os.path.dirname(__file__), "landmark.txt"), dtype=float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1, n_landmark, 2)
            self.std_landmark_anno = np.genfromtxt(os.path.join(os.path.dirname(__file__), "standard_landmark_68pts.txt"), dtype=float).reshape(n_landmark, 2)


    def __len__(self):
        return len(self.image_filenames)
    

    def apply_gaussian_blur(self, image):
        self.blur_sigma = 0.5
        if self.blur_sigma is not None and self.blur_sigma>0:
            return blur_face(image, self.blur_sigma)
        return image


    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        og = None

        if self.transform:
            image = self.transform(image)

        # if self.faceTransform:

        if self.crop == 'basic':
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
                og = image.copy_()
                image[:, y + padding:y + padding + h, x + padding:x + padding + w] = self.faceTransform(image[:, y + padding:y + padding + h, x + padding:x + padding + w])
            image = image[:, yp + padding : yp + padding + hp, xp + padding : xp + padding + wp]
            og = og[:, yp + padding : yp + padding + hp, xp + padding : xp + padding + wp]

            image = np.array(image.permute(1, 2, 0), dtype=np.uint8)
            image = cv2.resize(image, [self.dims, self.dims])
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # Normalize back to [0,1]

            og = np.array(og.permute(1, 2, 0), dtype=np.uint8)
            og = cv2.resize(og, [self.dims, self.dims])
            og = torch.from_numpy(og).permute(2, 0, 1).float()  # Normalize back to [0,1]

        elif self.crop == 'transform':
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

        elif self.crop == 'neural':
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
            # if self.faceTransform:
            #     image[:, y:y+h, x:x+w] = self.faceTransform(image[:, y:y+h, x:x+w])

            # do the landmark stuff
            image = align_crop_opencv(np.array(image.permute(1, 2, 0), dtype=np.uint8),  # Convert tensor to OpenCV format
                                            self.landmark_anno[int(filename[:-4])-1] * scale,  # Get source landmarks
                                            self.std_landmark_anno,
                                            crop_size=self.dims,
                                            face_factor=self.faceFactor)


            # blur the image 
            og = image.copy()
            image = self.apply_gaussian_blur(Image.fromarray(image, 'RGB'))

            # convert back to tensor
            og = torch.from_numpy(image).permute(2, 0, 1).float()
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # Normalize back to [0,1]
        
        if self.triplet: 
            target_id = self.identity[filename]
            same_id_images = [i for i, id_val in self.identity.items() if id_val == target_id and i != filename]

            if not same_id_images:
                #      anchor, blurred, id
                return og, image, target_id 
            else:
                random_filename = random.choice(same_id_images)

        else: 
            return image, self.identity[filename]


class CelebADual(): 
    def __init__(self, transform=None, faceTransform=None, dims=128, faceFactor=0.7, crop='neural', batch_size=32, shuffle=True):
        self.unBlurDataset = CelebADataset(transform=transform, faceTransform=faceTransform, dims=dims, faceFactor=faceFactor, crop=crop)
        self.BlurDataset = CelebADataset(transform=transform, faceTransform=None, dims=dims, faceFactor=faceFactor, crop=crop)

        # Create dataloaders for both versions
        self.unBlurLoader = DataLoader(self.unBlurDataset, batch_size=batch_size, shuffle=shuffle)
        self.BlurLoader = DataLoader(self.BlurDataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        # Zip the two dataloaders so they return corresponding batches
        return zip(iter(self.unBlurLoader), iter(self.BlurLoader))

class CelebATriple(): 
    def __init__(self):
        pass

    def __iter__(self):
        pass
    
