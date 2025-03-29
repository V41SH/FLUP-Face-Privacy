import torch
import cv2
from PIL import Image, ImageFilter
import numpy as np
import sys
from batch_face import RetinaFace
detector = RetinaFace(gpu_id=-1)

def detect_face(image):
    # Convert PIL image to cv2 format (RGB to BGR)
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Face detection
    faces = detector(img_cv, threshold=0.95, resize=1, max_size=1080, return_dict=True)
    return [face['box'] for face in faces]

# def blur_face(image, blur_sigma, blur_fn=None, faces=None):

def blur_face(image, blur_type='gaussian', blur_amount=3, **kwargs):
    faces = detect_face(image)
    
    if not faces:
        return image  # Return original image if no faces found
    
    result_img = image.copy()
    
    # x1 = 0
    # y1 = sys.maxsize
    # x2 = 0
    # y2 = sys.maxsize

    for face in faces:
        fx1, fy1, fx2, fy2 = face
        fx1, fy1 = max(0, int(fx1)), max(0, int(fy1))
        fx2, fy2 = min(image.width, int(fx2)), min(image.height, int(fy2))
        
        if fx2 > fx1 and fy2 > fy1:
            face_region = result_img.crop((fx1, fy1, fx2, fy2))
            
            if blur_type == 'gaussian':
                blurred_face = face_region.filter(ImageFilter.GaussianBlur(blur_amount))
            elif blur_type == 'black':
                blurred_face = Image.new("RGB", face_region.size, color=(0, 0, 0))
            elif blur_type == 'pixelation':
                pixel_size = kwargs.get('pixel_size', blur_amount)
                blurred_face = pixelate_face(face_region, pixel_size)
            else:
                raise ValueError(f"Unknown blur type: {blur_type}")
            
            result_img.paste(blurred_face, (fx1, fy1))
    
  
    return result_img

def pixelate_face(image_region, pixel_size=10):
    """
    Applies a pixelation effect by resizing down and up again.
    """
    width, height = image_region.size
    small = image_region.resize(
        (max(1, width // pixel_size), max(1, height // pixel_size)),
        resample=Image.NEAREST
    )
    return small.resize((width, height), Image.NEAREST)