import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from dataset import *
from tqdm import tqdm
# Define transformations (resize, normalize, convert to tensor)

def visualize_batch(batch_tensor):
    """
    Visualizes a batch of images (shape: [batch_size, 3, H, W]) using matplotlib.
    """
    # Ensure the tensor is in the correct format (detach from computation graph)
    # batch_tensor = batch_tensor.detach().cpu()
    # batch_tensor = torch.cat(batch_tensor[0], batch_tensor[2])

    # Create a grid of images (normalize=False ensures original pixel values)
    grid = vutils.make_grid(batch_tensor, nrow=8, normalize=True, scale_each=True)

    # Convert from (C, H, W) to (H, W, C) for Matplotlib
    grid = grid.permute(1, 2, 0).numpy()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "image.png")
    plt.imsave(save_path, grid)

    # Plot the images
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()

# Example usage with a random tensor (Replace with your actual batch)

face_transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size=15, sigma=(10, 20)),  # Apply Gaussian blur with random sigma
])

# Initialize dataset
# dataset = CelebADataset(faceTransform=face_transform, dims=128, faceFactor=0.7, basicCrop=False)
# # Create DataLoader
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

# # Check batch
# data_iter = iter(dataloader)
# sample_batch = next(data_iter)


dataset = CelebADual(faceTransform=face_transform, dims=128, faceFactor=0.7, basicCrop=True)

# for batch_tensor in dataset:
for (image_sharp, label_sharp), (image_blur, label_blur) in dataset:
    print(label_blur, label_sharp)
    batch_tensor = torch.cat([image_sharp, image_blur])
    visualize_batch(batch_tensor)
    break

# print("Batch shape:", np.array(sample_batch).size)  # Should be [64, 3, 128, 128]
