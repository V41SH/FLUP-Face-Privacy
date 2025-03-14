import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from dataset import *
# Define transformations (resize, normalize, convert to tensor)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128 (adjust as needed)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])
face_transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size=15, sigma=(10, 20)),  # Apply Gaussian blur with random sigma
])

def visualize_batch(batch_tensor):
    """
    Visualizes a batch of images (shape: [batch_size, 3, H, W]) using matplotlib.
    """
    # Ensure the tensor is in the correct format (detach from computation graph)
    batch_tensor = batch_tensor.detach().cpu()

    # Create a grid of images (normalize=False ensures original pixel values)
    grid = vutils.make_grid(batch_tensor, nrow=8, normalize=True, scale_each=True)

    print("here")
    # Convert from (C, H, W) to (H, W, C) for Matplotlib
    grid = grid.permute(1, 2, 0).numpy()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "image.png")
    plt.imsave(save_path, grid)
    
    print("here2")


    # # Plot the images
    # plt.figure(figsize=(10, 10))
    # plt.imshow(grid)
    # plt.axis("off")
    # plt.show()

# Example usage with a random tensor (Replace with your actual batch)


# Initialize dataset
img_dir = '../dataset/pictures/Img'
version = "img"
# version = "img_align"
# version = "img_align_png"
print("celeb")
dataset = CelebADataset(img_dir, faceTransform=face_transform, version=version)

# Create DataLoader
print("dload")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Check batch
print("pls")
data_iter = iter(dataloader)
sample_batch = next(data_iter)
print("Batch shape:", sample_batch.shape)  # Should be [64, 3, 128, 128]

print("bruh")
visualize_batch(sample_batch)