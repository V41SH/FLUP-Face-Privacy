import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.blurring_utils import blur_face


class LFWDataset(Dataset):
    """
    Dataset loader for Labeled Faces in the Wild (LFW) dataset from Kaggle
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
        self.image_paths = []
        self.labels = []
        self.names = []

        person_folders = os.listdir(self.people_dir)

        label_idx = 0
        label_map = {}

        for person in person_folders:
            person_dir = os.path.join(self.people_dir, person)
            if os.path.isdir(person_dir):
                person_images = os.listdir(person_dir)

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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        real_idx = self.indices[idx]

        img_path = self.image_paths[real_idx]
        image = Image.open(img_path).convert('RGB')

        # apply blur if specified
        image = self.apply_gaussian_blur(image)

        label = self.labels[real_idx]
        name = self.names[real_idx]

        if self.transform:
            image = self.transform(image)

        return image, label, name

    def get_class_name(self, label):
        """Return the name of the person for a given label"""
        return self.class_names.get(label, "Unknown")


def get_lfw_dataloaders(root_dir, batch_size=32, img_size=224, seed=42, blur_sigma=None):
    """
    Create train and test dataloaders for the LFW dataset

    Args:
        root_dir (string): Directory with the LFW dataset
        batch_size (int): Batch size for DataLoader
        img_size (int): Size to resize images to
        seed (int): Random seed for reproducibility

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
    train_dataset = LFWDataset(root_dir=root_dir, transform=train_transform, train=True, seed=seed, blur_sigma=blur_sigma)
    test_dataset = LFWDataset(root_dir=root_dir, transform=test_transform, train=False, seed=seed, blur_sigma=blur_sigma)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_dataset.num_classes


def visualize_batch(dataloader, class_names=None, num_images=5):
    """
    Visualize a batch of images from the dataloader

    Args:
        dataloader: PyTorch dataloader
        class_names: Dictionary mapping class indices to names
        num_images: Number of images to display
    """
    # Get a batch
    images, labels, names = next(iter(dataloader))

    # Unnormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = images * std + mean

    # Plot images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        label_text = names[i] if class_names is None else class_names.get(labels[i].item(), "Unknown")
        axes[i].set_title(f"{label_text}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Assuming LFW dataset is downloaded from Kaggle and extracted to 'data/lfw/'
    root_dir = 'data/lfw/'

    # Create dataloaders
    train_loader, test_loader, num_classes = get_lfw_dataloaders(
        root_dir,
        batch_size=32,
        blur_sigma=3
    )

    print(f"Dataset loaded successfully with {num_classes} unique individuals")
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Visualize a batch
    dataset = train_loader.dataset
    class_names = {i: dataset.get_class_name(i) for i in range(dataset.num_classes)}
    visualize_batch(train_loader, class_names)