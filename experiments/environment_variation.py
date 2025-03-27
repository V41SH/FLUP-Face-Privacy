from PIL import Image
import numpy as np

def progressive_crop(img, bounding_box, num_results):
    """
    Create a progressive series of crops from an image based on a bounding box.

    Args:
    img (PIL.Image): Input image
    bounding_box (tuple): Bounding box coordinates (x1, y1, x2, y2)
    num_results (int): Number of progressive crops to generate

    Returns:
    list: List of progressively cropped images
    """
    # Ensure input is valid
    if num_results < 1:
        raise ValueError("num_results must be at least 1")

    # Unpack bounding box
    x1, y1, x2, y2 = bounding_box

    # Calculate distances to image boundaries
    width, height = img.size

    # Calculate minimum distance to any boundary from the bounding box
    left_dist = x1
    right_dist = width - x2
    top_dist = y1
    bottom_dist = height - y2

    # Find the minimum distance
    min_dist = min(left_dist, right_dist, top_dist, bottom_dist)

    # Generate progressive crops
    results = []
    for i in range(1, num_results + 1):
        # Calculate crop amount
        crop_amount = (min_dist * i) // num_results

        # Determine crop coordinates
        new_x1 = max(0, x1 - crop_amount)
        new_y1 = max(0, y1 - crop_amount)
        new_x2 = min(width, x2 + crop_amount)
        new_y2 = min(height, y2 + crop_amount)

        # Crop the image
        cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
        results.append(cropped_img)

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