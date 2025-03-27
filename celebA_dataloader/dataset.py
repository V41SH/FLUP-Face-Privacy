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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from utils.blurring_utils import blur_face


class CelebADataset(Dataset):
    def __init__(self, transform=None, dims=128, faceFactor=0.7, triplet=False, blur_sigma=None, train=True, train_ratio=0.8, seed=42):
        self.faceFactor = faceFactor
        self.blur_sigma = blur_sigma
        self.dims = dims
        self.triplet = triplet
        self.transform = transform
        self.train_ratio = train_ratio

        self.img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../celebA/Img/img_celeba/"))
        self.image_filenames = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]

        with open(os.path.join(self.img_dir, "../../Anno/identity_CelebA.txt"), "r") as file:
            lines = file.readlines()
            self.identity = {line.split()[0]: int(line.split()[1]) for line in lines[2:]}

        with open(os.path.join(self.img_dir, "../../Anno/list_bbox_celeba.txt"), "r") as file:
            lines = file.readlines()
            self.bbox_anno = {line.split()[0]: list(map(int, line.split()[1:])) for line in lines[2:]}

        with open(os.path.join(os.path.dirname(__file__), "landmark.txt"), "r") as file:
            line = file.readline()
            n_landmark = len(re.split('[ ]+', line)[1:]) // 2

        self.landmark_anno = np.genfromtxt(os.path.join(os.path.dirname(__file__), "landmark.txt"), dtype=float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1, n_landmark, 2)
        self.std_landmark_anno = np.genfromtxt(os.path.join(os.path.dirname(__file__), "standard_landmark_68pts.txt"), dtype=float).reshape(n_landmark, 2)

        #train test splitting
        indices = np.arange(len(self.image_filenames))
        np.random.seed(seed)
        np.random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)

        if train:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]


    def __len__(self):
        return len(self.image_filenames)
    

    def apply_gaussian_blur(self, image):
        if self.blur_sigma is not None and self.blur_sigma>0:
            return blur_face(image, self.blur_sigma)
        return image
    
    def getFace(self, blurImg, filename): 

        # Get face bounding box
        x, y, w, h = self.bbox_anno[filename]

        # scale the image a little - ensure that the face is approximately the same size, 
        # according to the final crop size. assuming square crop size
        scale = self.dims / (h * 1.3) # to scale the image to approximately the same size as crop
        blurImg.thumbnail([int(blurImg.width * scale), int(blurImg.height* scale)])

        # Get face bounding box
        x = int(x * scale) 
        y = int(y * scale) 
        w = int(w * scale) 
        h = int(h * scale) 

        blurImg = align_crop_opencv(np.array(blurImg)[:, :, ::-1],  # Convert tensor to OpenCV format
                                        self.landmark_anno[int(filename[:-4])-1] * scale,  # Get source landmarks
                                        self.std_landmark_anno,
                                        crop_size=[blurImg.height, blurImg.width],
                                        face_factor=self.faceFactor)
        
        return Image.fromarray(blurImg, 'RGB')


    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        identity = self.identity[filename]

        blurImg = Image.open(img_path).convert("RGB")
        unblurImg = blurImg.copy()

        # center the eyes and the face in the middle of the image
        blurImg = self.getFace(blurImg, filename)
        # crop the image to the bounding box of the face + add blurring
        blurImg = self.apply_gaussian_blur(blurImg)

        if self.transform:
            blurImg = self.transform(blurImg)
    
        if self.triplet: 
            same_id_images = [i for i, id_val in self.identity.items() if id_val == identity and i != filename]

            if not same_id_images:
                return unblurImg, blurImg, identity 
            else:
                random_filename = random.choice(same_id_images)
                img_path = os.path.join(self.img_dir, random_filename)

                new_unblur = self.getFace(Image.open(img_path).convert("RGB"), random_filename)
                if self.transform: 
                    new_unblur = self.transform(new_unblur)
                return new_unblur, blurImg, identity 

        else: 
            return blurImg, self.identity[filename]


class CelebATriplet():
    def __init__(self, transforms=None, train=True, train_ratio=0.8, batch_size=32, img_size=224, seed=42, blur_sigma=None):
        self.data_dual = CelebADataset(transform=transforms, triplet=True, blur_sigma=blur_sigma,
                                       train=train, train_ratio=train_ratio, seed=seed)
        self.data_single = CelebADataset(transform=transforms, triplet=False, blur_sigma=blur_sigma,
                                         train=train, train_ratio=train_ratio, seed=seed)

        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.data_dual) // self.batch_size

    def __iter__(self):
        idx = 0
        while idx + self.batch_size <= len(self.data_dual):
            img1_batch = []
            img1_blur_batch = []
            img2_batch = []
            label_batch = []

            for _ in range(self.batch_size):
                img1, img1_blur, label1 = self.data_dual[idx]
                idx += 1

                # Find a negative sample with a different label
                while True:
                    idx2 = self.rng.randint(0, len(self.data_single) - 1)
                    img2, label2 = self.data_single[idx2]
                    if label2 != label1:
                        break

                img1_batch.append(img1)
                img1_blur_batch.append(img1_blur)
                img2_batch.append(img2)
                label_batch.append(label1)

            # Stack to form batches: [B, C, H, W] for images, [B] for labels
            yield (
                torch.stack(img1_batch),
                torch.stack(img1_blur_batch),
                torch.stack(img2_batch),
                torch.tensor(label_batch)
            )


    

def getCelebADataLoader(batch_size=32, img_size=224, seed=42, blur_sigma=None):

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train = CelebATriplet(transforms=train_transform, train=True, batch_size=batch_size, img_size=img_size, seed=seed, blur_sigma=blur_sigma)
    test = CelebATriplet(transforms=test_transform, train=False, batch_size=batch_size, img_size=img_size, seed=seed, blur_sigma=blur_sigma)

    return train, test


def visualize_batch(sample_batch, save_name="image.png"):
    """
    Visualizes a triplet batch: (img1, img1_blur, img2) side-by-side.
    Expects 3 images as PIL.Image or torch.Tensor in shape [3, H, W].
    """

    def to_tensor(img):
        if isinstance(img, Image.Image):
            return transforms.ToTensor()(img)
        return img

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    if len(sample_batch) == 3:
        img1, img1_blur, label = sample_batch

        # Convert all to tensors (if they aren’t already)
        img1 = to_tensor(img1)
        img1_blur = to_tensor(img1_blur)

        img1 = img1 * std + mean
        img1_blur = img1_blur * std + mean
        # img2 = to_tensor(img2)

        # Stack horizontally to make a row: [img1 | img1_blur | img2]
        row = torch.cat([img1, img1_blur], dim=2)  # concat along width
    if len(sample_batch) == 4:
        img1, img1_blur, img2, label = sample_batch

        # Convert all to tensors (if they aren’t already)
        img1 = to_tensor(img1)
        img1_blur = to_tensor(img1_blur)
        img2 = to_tensor(img2)

        img1 = img1 * std + mean
        img1_blur = img1_blur * std + mean
        img2 = img2 * std + mean

        # Stack horizontally to make a row: [img1 | img1_blur | img2]
        row = torch.cat([img1, img1_blur, img2], dim=2)  # concat along width

    # Optionally convert to grid shape: [1, 3, H, 3*W]
    grid = vutils.make_grid(row, normalize=True, scale_each=True)

    # Prepare for matplotlib
    grid_np = grid.permute(1, 2, 0).numpy()

    # Save and plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, save_name)
    plt.imsave(save_path, grid_np)

    plt.figure(figsize=(12, 4))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title(f"Triplet Visualization (label={label})")
    plt.show()


if __name__ == "__main__": 

    # face_transform = transforms.Compose([
    #     transforms.GaussianBlur(kernel_size=15, sigma=(10, 20)),  # Apply Gaussian blur with random sigma
    # ])
    # train_transform = transforms.Compose([
    #     transforms.Resize((200, 200)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # dataset = CelebADataset(triplet=True, transform=train_transform, seed=123, blur_sigma=7)

    # visualize_batch(dataset[0])

    # # Initialize dataset
    # dataset = CelebADataset(transform=train_transform)
    # # # Create DataLoader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    # data_iter = iter(dataloader)
    # sample_batch = next(data_iter)
    # visualize_batch(sample_batch)

    train, test = getCelebADataLoader(batch_size=1, blur_sigma=7, seed=123)
    data_iter = iter(train)
    sample_batch = next(data_iter)
    visualize_batch(sample_batch)

    # # Check batch


    # dataset = CelebADual(faceTransform=face_transform, dims=128, faceFactor=0.7, crop='neural')

    # for batch_tensor in dataset:
    # for (image_sharp, label_sharp), (image_blur, label_blur) in dataset:
    #     print(label_blur, label_sharp)
    #     batch_tensor = torch.cat([image_sharp, image_blur])
    #     visualize_batch(batch_tensor)
    #     break

    # print("Batch shape:", np.array(sample_batch).size)  # Should be [64, 3, 128, 128]