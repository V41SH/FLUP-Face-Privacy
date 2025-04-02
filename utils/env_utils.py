import cv2
import numpy as np
from PIL import Image
import random

import random
from PIL import Image
import torch

from utils.blurring_utils import detect_face, detector

def random_crop_pil(img, faces, cropping_steps):
    """
    Random crop using PIL without converting to OpenCV.
    Args:
        img (PIL.Image): Input image
        bounding_box (tuple): Bounding box coordinates (x1, y1, x2, y2)
        cropping_steps (int): Number of cropping steps.
        final_size (tuple): Dimensions (width, height) to resize all images to.
                            If None, all images are resized to the smallest cropped size.

    Returns:
        PIL.Image: Randomly cropped and resized image as PIL Image
    """
    if cropping_steps < 1:
        raise ValueError("cropping_steps must be at least 1")

    if not faces:
        return img

    # getting union of all faces
    
    #get largest face 
    bbox = max(faces, key=lambda box: abs(box[2]-box[0])*abs(box[3]-box[1]))

    x1, y1, x2, y2 = bbox

    width, height = img.size

    # Distance to edges
    left_dist = x1
    right_dist = width - x2
    top_dist = y1
    bottom_dist = height - y2
    min_dist = min(left_dist, right_dist, top_dist, bottom_dist)

    i = random.randint(1, cropping_steps)

    crop_amount = (min_dist * i) // cropping_steps

    new_x1 = max(0, x1 - crop_amount)
    new_y1 = max(0, y1 - crop_amount)
    new_x2 = min(width, x2 + crop_amount)
    new_y2 = min(height, y2 + crop_amount)

    crop = img.crop((new_x1, new_y1, new_x2, new_y2))

    return crop

def tight_crop_pil(img, faces=None, cropping_steps=1):
    """
    Random crop using PIL without converting to OpenCV.
    Args:
        img (PIL.Image): Input image
        bounding_box (tuple): Bounding box coordinates (x1, y1, x2, y2)
        cropping_steps (int): Number of cropping steps.
        final_size (tuple): Dimensions (width, height) to resize all images to.
                            If None, all images are resized to the smallest cropped size.

    Returns:
        PIL.Image: Randomly cropped and resized image as PIL Image
    """
    if cropping_steps < 1:
        raise ValueError("cropping_steps must be at least 1")

    if not faces:
        faces = detect_face(img)

    if not faces:
        return img

    # getting union of all faces
    
    #get largest face 
    bbox = max(faces, key=lambda box: abs(box[2]-box[0])*abs(box[3]-box[1]))

    x1, y1, x2, y2 = bbox

    width, height = img.size

    # Distance to edges
    left_dist = x1
    right_dist = width - x2
    top_dist = y1
    bottom_dist = height - y2
    min_dist = min(left_dist, right_dist, top_dist, bottom_dist)

    i = 0

    crop_amount = (min_dist * i) // cropping_steps

    new_x1 = max(0, x1 - crop_amount)
    new_y1 = max(0, y1 - crop_amount)
    new_x2 = min(width, x2 + crop_amount)
    new_y2 = min(height, y2 + crop_amount)

    crop = img.crop((new_x1, new_y1, new_x2, new_y2))

    return crop


def random_crop_cv(img, bounding_box, cropping_steps, final_size=None):
    """
    Based on progressive crop. Pick random.
    Args:
        img (PIL.Image): Input image
        bounding_box (tuple): Bounding box coordinates (x1, y1, x2, y2)
        num_results (int): Number of progressive crops to generate
        final_size (tuple): Dimensions (width, height) to resize all images to.
                            If None, all images are resized to the smallest cropped size.

    Returns:
        list: List of progressively cropped and resized images as PIL Images
    """
    if cropping_steps < 1:
        raise ValueError("cropping_steps must be at least 1")

    x1, y1, x2, y2 = bounding_box
    width, height = img.size

    # Distance to edges
    left_dist = x1
    right_dist = width - x2
    top_dist = y1
    bottom_dist = height - y2
    min_dist = min(left_dist, right_dist, top_dist, bottom_dist)

    img_cv = np.array(img.convert("RGB"))[:, :, ::-1]  # PIL to BGR (OpenCV)

    min_crop_shape = None

    # for i in range(1, cropping_steps + 1):

    i = random.randint(1, cropping_steps)

    crop_amount = (min_dist * i) // cropping_steps

    new_x1 = max(0, x1 - crop_amount)
    new_y1 = max(0, y1 - crop_amount)
    new_x2 = min(width, x2 + crop_amount)
    new_y2 = min(height, y2 + crop_amount)

    crop = img_cv[new_y1:new_y2, new_x1:new_x2]

    if final_size is None:
        # Keep track of the smallest shape
        h, w = crop.shape[:2]
        if min_crop_shape is None or (h * w < min_crop_shape[0] * min_crop_shape[1]):
            min_crop_shape = (w, h)

    # Resize all crops to the target size
    target_size = final_size if final_size else min_crop_shape
    # resized = [cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR) for crop in crops]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert back to PIL
    results = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    return results



def progressive_crop(img, bounding_box, num_results, final_size=None):
    """
    Create a progressive series of crops from an image based on a bounding box.

    Args:
        img (PIL.Image): Input image
        bounding_box (tuple): Bounding box coordinates (x1, y1, x2, y2)
        num_results (int): Number of progressive crops to generate
        final_size (tuple): Dimensions (width, height) to resize all images to.
                            If None, all images are resized to the smallest cropped size.

    Returns:
        list: List of progressively cropped and resized images as PIL Images
    """
    if num_results < 1:
        raise ValueError("num_results must be at least 1")

    x1, y1, x2, y2 = bounding_box
    width, height = img.size

    # Distance to edges
    left_dist = x1
    right_dist = width - x2
    top_dist = y1
    bottom_dist = height - y2
    min_dist = min(left_dist, right_dist, top_dist, bottom_dist)

    img_cv = np.array(img.convert("RGB"))[:, :, ::-1]  # PIL to BGR (OpenCV)

    crops = []
    min_crop_shape = None

    for i in range(1, num_results + 1):
        crop_amount = (min_dist * i) // num_results

        new_x1 = max(0, x1 - crop_amount)
        new_y1 = max(0, y1 - crop_amount)
        new_x2 = min(width, x2 + crop_amount)
        new_y2 = min(height, y2 + crop_amount)

        crop = img_cv[new_y1:new_y2, new_x1:new_x2]

        if final_size is None:
            # Keep track of the smallest shape
            h, w = crop.shape[:2]
            if min_crop_shape is None or (h * w < min_crop_shape[0] * min_crop_shape[1]):
                min_crop_shape = (w, h)

        crops.append(crop)

    # Resize all crops to the target size
    target_size = final_size if final_size else min_crop_shape
    resized = [cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR) for crop in crops]

    # Convert back to PIL
    results = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in resized]
    return results

# Example usage and visualization
def visualize_progressive_crops(img, bounding_box, num_results):
    """
    Visualize the progressive crops
    """
    import matplotlib.pyplot as plt

    # Generate progressive crops
    crops = progressive_crop(img, bounding_box, num_results)

    # Create visualization
    fig, axes = plt.subplots(1, num_results, figsize=(15, 5))

    for i, crop in enumerate(crops):
        axes[i].imshow(crop)
        axes[i].set_title(f'Crop {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return crops


# Demonstration script
if __name__ == "__main__":
    # Example usage
    # Load an example image (replace with your own)
    from PIL import Image

    # Example image and bounding box
    img = Image.open('../data/lfw/lfw-deepfunneled/lfw-deepfunneled/Al_Gore/Al_Gore_0001.jpg')

    # Example bounding box (x1, y1, x2, y2)
    bounding_box = (100, 100, 300, 300)

    # Visualize 5 progressive crops
    visualize_progressive_crops(img, bounding_box, num_results=5)


# if __name__ == "__main__":
#     from PIL import Image
#     import os

#     # Path to an image from the img_celeba folder
#     # current_dir = os.path.dirname(os.path.abspath(__file__))
#     # img_path = os.path.join(current_dir, "..", "celebA", "Img", "img_celeba", "000001.jpg")
#     img_path = "img2.png"

#     # Load the image
#     img = Image.open(img_path)

#     # Example bounding box (x1, y1, x2, y2)

#     bounding_box = (95, 71, 321, 384)  # Adjust this depending on the image

#     # Visualize 5 progressive crops
# #     visualize_progressive_crops(img, bounding_box, num_results=5)
