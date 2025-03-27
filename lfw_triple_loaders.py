from xml.sax.handler import all_properties

import torch
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.blurring_utils import blur_face
from utils.lfw_utils import *

class LFWDatasetTriple(Dataset):
    """
    Dataset loader for triplets; Labeled Faces in the Wild (LFW) dataset from Kaggle
    """

    def __init__(self, root_dir, csv_file=None, transform=None, train=True, train_ratio=0.8, seed=42
                 , blur_sigma=3):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string, optional): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): Whether to load train or test split
            train_ratio (float): Ratio of data to use for training
            seed (int): Random seed for reproducibility
            blur_sigma (float): Blurring sigma for anchor
        """
        self.root_dir = root_dir
        self.transform = transform
        #self.anchor_blur = anchor_blur
        self.blur_sigma = blur_sigma

        # Set up paths
        self.people_dir = os.path.join(root_dir, 'lfw-deepfunneled', 'lfw-deepfunneled')

        # Get all image paths and labels
        self.all_people = []
        self.image_paths = [] # only bros with more than one pic
        self.labels = []
        self.names = []

        person_folders = os.listdir(self.people_dir)

        label_idx = 0
        label_map = {}

        for person in person_folders:
            person_dir = os.path.join(self.people_dir, person)
            if os.path.isdir(person_dir):
                person_images = os.listdir(person_dir)

                # we want bros with more than one pic
                if len(person_images) > 1:
                    if person not in label_map:
                        label_map[person] = label_idx
                        label_idx += 1

                    if person not in self.all_people:
                        self.all_people.append(person)

                    for img_name in person_images:
                        if img_name.endswith('.jpg'):
                            self.image_paths.append(os.path.join(person_dir, img_name))
                            self.labels.append(label_map[person])
                            self.names.append(person)

        # Train/test split
        indices = np.arange(len(self.image_paths))
        np.random.seed(seed)
        np.random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)

        if train:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        # Store number of classes
        self.num_classes = len(label_map)
        self.class_names = {v: k for k, v in label_map.items()}

    def apply_gaussian_blur(self, image):
        if self.blur_sigma is not None and self.blur_sigma>0:
            return blur_face(image, self.blur_sigma)
        return image

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        # print("TRIPLE STARTED")

        if torch.is_tensor(idx):
            idx = idx.tolist()

        real_idx = self.indices[idx]
        label = self.labels[real_idx]
        name = self.names[real_idx]

        anchor_path_1 = self.image_paths[real_idx]
        anchor_path_2 = get_diff_person(anchor_path_1, self.people_dir, self.all_people)
        positive_path_1 = get_same_person(anchor_path_1)
        positive_path_2 = get_same_person(anchor_path_2)

        anchor_1_sharp = Image.open(anchor_path_1)
        anchor_2_blur = self.apply_gaussian_blur(Image.open(anchor_path_2).convert('RGB'))
        positive_1_blur = self.apply_gaussian_blur(Image.open(positive_path_1).convert('RGB'))
        positive_2_sharp = Image.open(positive_path_2)

        # uncomment to test
        # anchor_1_sharp.show()
        # anchor_2_blur.show()
        # positive_1_blur.show()
        # positive_2_sharp.show()

        if self.transform:
            anchor_1_sharp = self.transform(anchor_1_sharp)
            anchor_2_blur = self.transform(anchor_2_blur)
            positive_1_blur = self.transform(positive_1_blur)
            positive_2_sharp = self.transform(positive_2_sharp)

        # print("BY SOME MIRACLE FINALIZED GETTING THE PROMISED SHIT")
        return anchor_1_sharp, anchor_2_blur, positive_1_blur, positive_2_sharp

    def get_class_name(self, label):
        """Return the name of the person for a given label"""
        return self.class_names.get(label, "Unknown")


def get_lfw_dataloaders(root_dir, batch_size=32, img_size=224, seed=42,
                        anchor_blur = False, blur_sigma=None):
    """
    Create train and test dataloaders for the LFW dataset

    Args:
        root_dir (string): Directory with the LFW dataset
        batch_size (int): Batch size for DataLoader
        img_size (int): Size to resize images to
        seed (int): Random seed for reproducibility
        anchor_blur (bool): Whether to you want blurred anchor and sharp positive and negative
        blur_sigma (float): Blurring sigma for anchor

    Returns:
        train_loader, test_loader, num_classes
    """
    # Define transformations
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

    # Create datasets
    train_dataset = LFWDatasetTriple(root_dir=root_dir, transform=train_transform, train=True, seed=seed,
                                    blur_sigma=blur_sigma)
    test_dataset = LFWDatasetTriple(root_dir=root_dir, transform=test_transform, train=False, seed=seed,
                                    blur_sigma=blur_sigma)

    # Create dataloaders
    #NUM_WORKERS=4
    NUM_WORKERS=0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, test_loader, train_dataset.num_classes


if __name__ == "__main__":
    # Assuming LFW dataset is downloaded from Kaggle and extracted to 'data/lfw/'
    root_dir = 'data/lfw/'

    # Create dataloaders
    train_loader, test_loader, num_classes = get_lfw_dataloaders(
        root_dir,
        batch_size=1,
        #anchor_blur=False,
        blur_sigma=3
    )

    print(f"Dataset loaded successfully with {num_classes} unique individuals")
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")


    im_1, im_2, im_3, im_4, _, _, _ = train_loader.dataset[0]