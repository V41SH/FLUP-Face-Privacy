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

from utils.blurring_utils import detect_face, blur_face
from utils.lfw_utils import *
from utils.env_utils import random_crop_pil as random_crop
import pickle

class LFWDatasetTriple(Dataset):
    """
    Dataset loader for triplets; Labeled Faces in the Wild (LFW) dataset from Kaggle
    """

    def __init__(self, root_dir, csv_file=None, transform=None, train=True, train_ratio=0.8, seed=42
                 , blur_sigma=3, randomize_blur=False, randomize_crop=False, preload_bboxes=True):
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
        self.blur_sigma = blur_sigma
        self.randomize_blur = randomize_blur
        self.randomize_crop = randomize_crop
        self.preload_bboxes = preload_bboxes
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


        if self.preload_bboxes:
            with open('result_bbox.pickle', 'rb') as handle:
                self.bboxes = pickle.load(handle)

    def apply_gaussian_blur(self, image, faces=None):

        if self.randomize_blur:
            # then self.blur_sigma is list of two values?
            if type(self.blur_sigma) == list:
                minsig, maxsig = self.blur_sigma[0], self.blur_sigma[1]
            
            else:
                # default values
                minsig, maxsig = 5 , 20

            sigma = random.randint(minsig, maxsig)
            print(sigma)
            return blur_face(image, blur_type='gaussian', blur_amount=sigma, faces=faces)

        else:
            if self.blur_sigma is not None and self.blur_sigma>0:
                return blur_face(image, blur_type='gaussian', blur_amount=self.blur_sigma, faces=faces)

        return image

    def apply_crop(self, image, faces=None):
        
        if self.randomize_crop:
            image = random_crop(image, faces=faces, cropping_steps=10)

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

        image_anchor_path_1 = Image.open(anchor_path_1) 
        image_anchor_path_2 = Image.open(anchor_path_2) 
        image_positive_path_1 = Image.open(positive_path_1) 
        image_positive_path_2  = Image.open(positive_path_2)
        
        if self.preload_bboxes:
            # print("root_dir", self.root_dir)
            ap1 = anchor_path_1[anchor_path_1.find(root_dir):]
            ap2 = anchor_path_2[anchor_path_2.find(root_dir):]
            ip1 = positive_path_1[positive_path_1.find(root_dir):]
            ip2 = positive_path_2[positive_path_2.find(root_dir):]

            faces_anchor_path_1 = self.bboxes[ap1]
            faces_anchor_path_2 = self.bboxes[ap2]
            faces_positive_path_1 = self.bboxes[ip1]
            faces_positive_path_2 = self.bboxes[ip2]

        else:
            faces_anchor_path_2 = detect_face(image_anchor_path_2) 
            faces_positive_path_1 = detect_face(image_positive_path_1) 
            
            if self.randomize_crop:
                faces_anchor_path_1 = detect_face(image_anchor_path_1) 
                faces_positive_path_2  = detect_face(image_positive_path_2)



        anchor_1_sharp = image_anchor_path_1
        anchor_2_blur = self.apply_gaussian_blur(image_anchor_path_2, faces_anchor_path_2)
        positive_1_blur = self.apply_gaussian_blur(image_anchor_path_1, faces_positive_path_1)
        positive_2_sharp = image_positive_path_2

        if self.randomize_crop:
            anchor_1_sharp = self.apply_crop(anchor_1_sharp, faces_anchor_path_1)
            anchor_2_blur = self.apply_crop(anchor_2_blur, faces_anchor_path_2)
            positive_1_blur = self.apply_crop(positive_1_blur, faces_positive_path_1)
            positive_2_sharp = self.apply_crop(positive_2_sharp, faces_positive_path_2)

        # uncomment to test
        anchor_1_sharp.show()
        anchor_2_blur.show()
        positive_1_blur.show()
        positive_2_sharp.show()

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

def get_transforms(img_size):

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

    return train_transform, test_transform

def get_lfw_dataloaders(root_dir, batch_size=32, img_size=224, seed=42,
                        anchor_blur = False, blur_sigma=None, randomize_blur=False,
                        randomize_crop=False, preload_bboxes=True):
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
    # get transform
    train_transform, test_transform = get_transforms(img_size=img_size)
    # Create datasets
    train_dataset = LFWDatasetTriple(root_dir=root_dir, transform=train_transform, train=True, seed=seed,
                                    blur_sigma=blur_sigma, randomize_blur=randomize_blur, randomize_crop=randomize_crop,
                                    preload_bboxes=preload_bboxes)
    test_dataset = LFWDatasetTriple(root_dir=root_dir, transform=test_transform, train=False, seed=seed,
                                    blur_sigma=blur_sigma, randomize_blur=randomize_blur, randomize_crop=randomize_crop,
                                    preload_bboxes=preload_bboxes)

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
        # blur_sigma=3,
        blur_sigma=[5,20],
        randomize_blur=True,
        # randomize_crop=True,
        preload_bboxes=True
    )

    print(f"Dataset loaded successfully with {num_classes} unique individuals")
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")


    # im_1, im_2, im_3, im_4, _, _, _ = train_loader.dataset[0]
    im_1, im_2, im_3, im_4 = train_loader.dataset[0]