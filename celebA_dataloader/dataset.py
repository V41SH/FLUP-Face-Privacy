import os
import sys
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

# Set up project path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from cropper import *
from utils.blurring_utils import *


class CelebADataset(Dataset):
    def __init__(self, transform=None, dims=128, faceFactor=0.5, triplet=False, blur_sigma=None, train=True, train_ratio=0.8, 
                 seed=42, blur_fn=None, anchor_blur=False, same_person=False, blur_both=False):
        self.faceFactor = faceFactor
        self.blur_sigma = blur_sigma
        self.dims = dims
        self.triplet = triplet
        self.transform = transform
        self.train_ratio = train_ratio
        self.blur_fn = blur_fn
        self.anchor_blur = anchor_blur
        self.same_person = same_person
        self.blur_both = blur_both

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
        if isinstance(image, torch.Tensor):
            image = Image.fromarray(np.array(image)[:, :, ::-1], "RGB")

        if (self.blur_sigma is not None and self.blur_sigma > 0) or self.blur_fn is not None:
            res = blur_face(image, self.blur_sigma, blur_fn=self.blur_fn) 
            return res 

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

        d = min(blurImg.height, blurImg.width)

        blurImg = align_crop_opencv(np.array(blurImg)[:, :, ::-1],  # Convert tensor to OpenCV format
                                        self.landmark_anno[int(filename[:-4])-1] * scale,  # Get source landmarks
                                        self.std_landmark_anno,
                                        crop_size=[d,d],
                                        face_factor=self.faceFactor)
        
        return Image.fromarray(blurImg[:, :, ::-1], "RGB")

    
    def blur_and_transform(self, anchor, positive=None): 

        #anchor_blur T F F T N
        #blur_both   T F T F N
        #num blured  2 1 2 1 0
        #which image   p   a

        if self.anchor_blur is None and self.blur_both is None: 
            if positive: 
                if self.transform: 
                    return self.transform(anchor), self.transform(positive)
                else: 
                    return anchor, positive
            else: 
                if self.transform: 
                    return self.transform(anchor)
                else: 
                    return anchor

        if self.anchor_blur and self.blur_both:
            anchor = self.apply_gaussian_blur(anchor)
            if positive: 
                positive = self.apply_gaussian_blur(positive)
                if self.transform: 
                    return self.transform(anchor), self.transform(positive)
                else: 
                    return anchor, positive
            else: 
                if self.transform: 
                    return self.transform(anchor)
                else: 
                    return anchor
        elif not self.anchor_blur and not self.blur_both:
            if positive: 
                positive = self.apply_gaussian_blur(positive)
                if self.transform:
                    return self.transform(anchor), self.transform(positive)
                else:
                    return anchor, positive
            else:
                if self.transform:
                    return self.transform(anchor)
                else:
                    return anchor 
        elif not self.anchor_blur and self.blur_both:
            print("invlid combination of anchor_blur and blur_both, bluring nothing")
            anchor = self.apply_gaussian_blur(anchor)
            if positive: 
                positive = self.apply_gaussian_blur(positive)
            
                if self.transform: 
                    return self.transform(anchor), self.transform(positive)
                else: 
                    return anchor, positive
            else: 
                if self.transform: 
                    return self.transform(anchor)
                else: 
                    return anchor

        elif self.anchor_blur and not self.blur_both:
            anchor = self.apply_gaussian_blur(anchor)
            if positive: 
                if self.transform: 
                    return self.transform(anchor), self.transform(positive)
                else: 
                    return anchor, positive
            else: 
                if self.transform: 
                    return self.transform(anchor)
                else: 
                    return anchor
        else: 
            print("this is undocumented behvaiour uwu, I will self distruct")
            return None



    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        identity = self.identity[filename]

        anchor = Image.open(img_path).convert("RGB")


        positive = anchor.copy()

        # center the eyes and the face in the middle of the image
        anchor = self.getFace(anchor, filename)
    
        if self.triplet or self.blur_both: 
            if self.same_person:
                same_id_images = [i for i, id_val in self.identity.items() if id_val == identity and i != filename]
            else: 
                same_id_images = [i for i, id_val in self.identity.items() if id_val != identity and i != filename]


            if not same_id_images:
                return *self.blur_and_transform(anchor, positive=positive), identity 

            else:
                random_filename = random.choice(same_id_images)
                img_path = os.path.join(self.img_dir, random_filename)

                random_positive = self.getFace(Image.open(img_path).convert("RGB"), random_filename)

                return *self.blur_and_transform(anchor, positive=random_positive), identity 

        else: 
            return self.blur_and_transform(anchor), identity 


class CelebATriplet():
    def __init__(self, transforms=None, train=True, train_ratio=0.8, batch_size=32, img_size=224, seed=42, blur_sigma=None, blur_fn=None, anchor_blur=False, same_person=False, blur_both=False):
        self.data_dual = CelebADataset(transform=transforms, triplet=True, blur_sigma=blur_sigma,
                                       train=train, train_ratio=train_ratio, seed=seed, blur_fn=blur_fn,
                                       anchor_blur=anchor_blur, same_person=same_person, blur_both=blur_both)

        self.data_single = CelebADataset(transform=transforms, triplet=False, blur_sigma=blur_sigma,
                                         train=train, train_ratio=train_ratio, seed=seed, blur_fn=blur_fn,
                                         anchor_blur=not anchor_blur, same_person=same_person, blur_both=blur_both)

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
                    # print(label1, label2)
                    if label2 != label1:
                        break

                img1_batch.append(torch.tensor(np.array(img1)))
                img1_blur_batch.append(torch.tensor(np.array(img1_blur)))
                img2_batch.append(torch.tensor(np.array(img2)))
                label_batch.append(torch.tensor(np.array(label1)))

            # Stack to form batches: [B, C, H, W] for images, [B] for labels
            yield (
                torch.stack(img1_batch),
                torch.stack(img1_blur_batch),
                torch.stack(img2_batch),
                torch.tensor(label_batch)
            )



def getCelebADataLoader(batch_size=32, img_size=224, seed=42, blur_sigma=None, blur_fn=None, anchor_blur=False, same_person=False, blur_both=False):

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

    train = CelebATriplet(transforms=train_transform, train=True, batch_size=batch_size, img_size=img_size, seed=seed, blur_sigma=blur_sigma, blur_fn=blur_fn,
                          anchor_blur=anchor_blur, same_person=same_person, blur_both=blur_both)
    test = CelebATriplet(transforms=test_transform, train=False, batch_size=batch_size, img_size=img_size, seed=seed, blur_sigma=blur_sigma, blur_fn=blur_fn,
                          anchor_blur=anchor_blur, same_person=same_person, blur_both=blur_both)

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

    if len(sample_batch) == 2: 
        img1, label = sample_batch
        img1 = to_tensor(img1)
        img1 = img1 * std + mean

        row = img1 
    elif len(sample_batch) == 3:
        img1, img1_blur, label = sample_batch

        img1 = to_tensor(img1)
        img1_blur = to_tensor(img1_blur)

        img1 = img1 * std + mean
        img1_blur = img1_blur * std + mean

        row = torch.cat([img1, img1_blur], dim=2)  # concat along width
    elif len(sample_batch) == 4:
        img1, img1_blur, img2, label = sample_batch

        img1 = to_tensor(img1)
        img1_blur = to_tensor(img1_blur)
        img2 = to_tensor(img2)

        img1 = img1 * std + mean
        img1_blur = img1_blur * std + mean
        img2 = img2 * std + mean

        print(img1.shape, img1_blur.shape, img2.shape, label)

        row = torch.cat([img1, img1_blur, img2], dim=2)  # concat along width
    else:
        label = "single image"
        row = to_tensor(sample_batch[0])

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
    img_size = 200
    seed = 5358
    blur_sigma = 7

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Triplet: anchor, blurred positive, and blurred negative
    train_loader, _ = getCelebADataLoader(
        batch_size=1,
        img_size=img_size,
        seed=seed,
        blur_sigma=blur_sigma,
        blur_fn=None
    )
    sample_batch = next(iter(train_loader))
    visualize_batch(sample_batch, save_name="3_1.png")

    # Triplet: blurred anchor, positive, and negative
    train_loader, _ = getCelebADataLoader(
        batch_size=1,
        img_size=img_size,
        seed=seed,
        blur_sigma=blur_sigma,
        blur_fn=None
    )
    # Manually set anchor_blur=True on the internal dataset
    train_loader.data_dual.anchor_blur = True
    sample_batch = next(iter(train_loader))
    visualize_batch(sample_batch, save_name="3_2.png")

    # Dual: same person, both are blurred
    dual_dataset = CelebADataset(
        triplet=True,
        transform=transform,
        seed=seed,
        blur_sigma=blur_sigma,
        same_person=True,
        blur_both=True
    )
    visualize_batch(dual_dataset[0], save_name="2_1.png")

    # Dual: same person, neither are blurred
    dual_dataset = CelebADataset(
        triplet=True,
        transform=transform,
        seed=seed,
        blur_sigma=blur_sigma,
        same_person=True,
        blur_both=None,
        anchor_blur=None
    )
    visualize_batch(dual_dataset[0], save_name="2_2.png")

    # Single: one blurred face
    single_dataset = CelebADataset(
        triplet=False,
        transform=transform,
        seed=seed,
        blur_sigma=blur_sigma,
        anchor_blur=True
    )
    visualize_batch(single_dataset[0], save_name="1_1.png")

    # Single: one normal face
    single_dataset = CelebADataset(
        triplet=False,
        transform=transform,
        seed=seed,
        blur_sigma=0
    )
    visualize_batch(single_dataset[0], save_name="1_2.png")

    # Single: one pixelated face
    single_dataset = CelebADataset(
        triplet=False,
        transform=transform,
        seed=seed,
        # blur_sigma=0,
        blur_fn=pixelation_blur_fn, 
        anchor_blur=True
    )
    visualize_batch(single_dataset[0], save_name="1_3.png")

    # Single: one blacked out face
    single_dataset = CelebADataset(
        triplet=False,
        transform=transform,
        seed=seed,
        # blur_sigma=0,
        blur_fn=black_blur_fn,
        anchor_blur=True
    )
    visualize_batch(single_dataset[0], save_name="1_4.png")