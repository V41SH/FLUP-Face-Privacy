import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.blurring_utils import blur_face

class LFWDatasetTripleSharpAnchor(Dataset):
    """
    Dataset loader for triplets; Labeled Faces in the Wild (LFW) dataset from Kaggle
    """

    def __init__(self, root_dir, csv_file=None, transform=None, train=True, train_ratio=0.8, seed=42, blur_sigma=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string, optional): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): Whether to load train or test split
            train_ratio (float): Ratio of data to use for training
            seed (int): Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.transform = transform
        self.blur_sigma = blur_sigma

        # Set up paths
        self.people_dir = os.path.join(root_dir, 'lfw-deepfunneled', 'lfw-deepfunneled')

        # Get all image paths and labels
        self.all_people = os.listdir(self.people_dir)
        self.image_paths = [] # only with bros with more than one pic
        self.labels = []
        self.names = []

        person_folders = os.listdir(self.people_dir)

        label_idx = 0
        label_map = {}

        for person in person_folders:
            person_dir = os.path.join(self.people_dir, person)
            if os.path.isdir(person_dir):
                person_images = os.listdir(person_dir)
                #self.all_images.extend(person_images)

                # we want bros with more than one pics
                if len(person_images) > 1:
                    if person not in label_map:
                        label_map[person] = label_idx
                        label_idx += 1

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

        print("STARTED TO TRY TO MAYBE SOMETIMES BY CHANCE GET ITEM")

        if torch.is_tensor(idx):
            idx = idx.tolist()

        real_idx = self.indices[idx]
        label = self.labels[real_idx]
        name = self.names[real_idx]

        img_path = self.image_paths[real_idx]
        image = Image.open(img_path).convert('RGB')

        # apply blur for same person, different image
        image_folder_path = os.path.abspath(os.path.join(img_path, os.pardir))
        image_list = os.listdir(image_folder_path)
        image_list.remove(os.path.basename(img_path)) # cause we dont want to give the same img lol
        image_same_path = random.choice(image_list)
        image_same_path = os.path.join(image_folder_path, image_same_path)
        
        image_same_blur = self.apply_gaussian_blur(Image.open(image_same_path))

        # pick random pic and blur that boiiiii
        self.all_people.remove(os.path.basename(image_folder_path))
        rando_person = random.choice(self.all_people)
        self.all_people.append(os.path.basename(image_folder_path))
        rando_person_path = os.path.join(self.people_dir, rando_person)
        rando_image = random.choice(os.listdir(rando_person_path))
        rando_imag_blur = self.apply_gaussian_blur(Image.open(os.path.join(rando_person_path, rando_image)))


        if self.transform:
            image = self.transform(image)

        print("BY SOME MIRACLE FINALIZED GETTING THE PROMISED SHIT")
        return image, image_same_blur, rando_imag_blur

    def get_class_name(self, label):
        """Return the name of the person for a given label"""
        return self.class_names.get(label, "Unknown")


class LFWDatasetTripleBlurAnchor(Dataset):
    """
    Dataset loader for triplets; Labeled Faces in the Wild (LFW) dataset from Kaggle
    """

    def __init__(self, root_dir, csv_file=None, transform=None, train=True, train_ratio=0.8, seed=42, blur_sigma=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string, optional): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): Whether to load train or test split
            train_ratio (float): Ratio of data to use for training
            seed (int): Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.transform = transform
        self.blur_sigma = blur_sigma

        # Set up paths
        self.people_dir = os.path.join(root_dir, 'lfw-deepfunneled', 'lfw-deepfunneled')

        # Get all image paths and labels
        self.all_people = os.listdir(self.people_dir)
        self.image_paths = []  # only with bros with more than one pic
        self.labels = []
        self.names = []

        person_folders = os.listdir(self.people_dir)

        label_idx = 0
        label_map = {}

        for person in person_folders:
            person_dir = os.path.join(self.people_dir, person)
            if os.path.isdir(person_dir):
                person_images = os.listdir(person_dir)
                # self.all_images.extend(person_images)

                # we want bros with more than one pics
                if len(person_images) > 1:
                    if person not in label_map:
                        label_map[person] = label_idx
                        label_idx += 1

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
        if self.blur_sigma is not None and self.blur_sigma > 0:
            return blur_face(image, self.blur_sigma)
        return image

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        print("STARTED TO TRY TO MAYBE SOMETIMES BY CHANCE GET ITEM")

        if torch.is_tensor(idx):
            idx = idx.tolist()

        real_idx = self.indices[idx]
        label = self.labels[real_idx]
        name = self.names[real_idx]

        img_path = self.image_paths[real_idx]
        image = Image.open(img_path).convert('RGB')
        image = self.apply_gaussian_blur(image)

        # apply blur for same person, different image
        image_folder_path = os.path.abspath(os.path.join(img_path, os.pardir))
        image_list = os.listdir(image_folder_path)
        image_list.remove(os.path.basename(img_path))  # cause we dont want to give the same img lol
        image_same_path = random.choice(image_list)
        image_same_path = os.path.join(image_folder_path, image_same_path)

        image_same_blur = Image.open(image_same_path)

        # pick random pic and blur that boiiiii
        self.all_people.remove(os.path.basename(image_folder_path))
        rando_person = random.choice(self.all_people)
        self.all_people.append(os.path.basename(image_folder_path))
        rando_person_path = os.path.join(self.people_dir, rando_person)
        rando_image = random.choice(os.listdir(rando_person_path))
        rando_imag_blur = Image.open(os.path.join(rando_person_path, rando_image))

        if self.transform:
            image = self.transform(image)

        print("BY SOME MIRACLE FINALIZED GETTING THE PROMISED SHIT")
        return image, image_same_blur, rando_imag_blur

    def get_class_name(self, label):
        """Return the name of the person for a given label"""
        return self.class_names.get(label, "Unknown")

def get_lfw_dataloaders(root_dir, batch_size=32, img_size=224, seed=42, anchor_blur = False, blur_sigma=None):
    """
    Create train and test dataloaders for the LFW dataset

    Args:
        root_dir (string): Directory with the LFW dataset
        batch_size (int): Batch size for DataLoader
        img_size (int): Size to resize images to
        seed (int): Random seed for reproducibility
        anchor_blur (bool): Whether to you want blurred anchor and sharp positive and negative

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
    if anchor_blur:
        train_dataset = LFWDatasetTripleBlurAnchor(root_dir=root_dir, transform=train_transform, train=True, seed=seed,
                                                    blur_sigma=blur_sigma)
        test_dataset = LFWDatasetTripleBlurAnchor(root_dir=root_dir, transform=test_transform, train=False, seed=seed,
                                                   blur_sigma=blur_sigma)
    else:
        train_dataset = LFWDatasetTripleSharpAnchor(root_dir=root_dir, transform=train_transform, train=True, seed=seed,
                                                    blur_sigma=blur_sigma)
        test_dataset = LFWDatasetTripleSharpAnchor(root_dir=root_dir, transform=test_transform, train=False, seed=seed,
                                                   blur_sigma=blur_sigma)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_dataset.num_classes