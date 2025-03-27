import torch
# from insightface.app import FaceAnalysis
import cv2
from PIL import Image, ImageFilter
import numpy as np

from batch_face import RetinaFace
detector = RetinaFace(gpu_id=0)


# def detect_face(image):
#     # Initialize InsightFace FaceAnalysis
#     # face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])

#     face_analyzer = FaceAnalysis(providers=providers)
#     face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

#     # Convert PIL image to cv2 format for InsightFace
#     img_np = np.array(image)
#     img_cv = img_np[:, :, ::-1].copy()  # RGB to BGR

#     # Detect faces using InsightFace
#     faces = face_analyzer.get(img_cv)

#     return faces


# def blur_face(image, blur_sigma, blur_fn=None):
#     faces = detect_face(image)

#     # If no faces then return the original image
#     if faces is None or len(faces) == 0:
#         return image

#     result_img = image.copy()

#     for face in faces:
#         bbox = face.bbox.astype(int)
#         x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(image.width, x2), min(image.height, y2)

#         if x2 > x1 and y2 > y1:
#             face_region = result_img.crop((x1, y1, x2, y2))

#             # Apply custom blur function if provided
#             if blur_fn is not None:
#                 blurred_face = blur_fn(face_region)
#             else:
#                 blurred_face = face_region.filter(ImageFilter.GaussianBlur(blur_sigma))

#             result_img.paste(blurred_face, (x1, y1))

#     return result_img
def detect_face(image):
    detector = RetinaFace(gpu_id=0)
    
    # Convert PIL image to cv2 format (RGB to BGR)
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Face detection
    faces = detector(img_cv, threshold=0.95, resize=1, max_size=1080, return_dict=True)
    return faces

def blur_face(image, blur_sigma, blur_fn=None):
    faces = detect_face(image)
    
    if not faces:
        return image  # Return original image if no faces found
    
    result_img = image.copy()
    
    for face in faces:
        x1, y1, x2, y2 = face['box']
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
        
        if x2 > x1 and y2 > y1:
            face_region = result_img.crop((x1, y1, x2, y2))
            
            # Apply custom blur function if provided
            if blur_fn is not None:
                blurred_face = blur_fn(face_region)
            else:
                blurred_face = face_region.filter(ImageFilter.GaussianBlur(blur_sigma))
            
            result_img.paste(blurred_face, (x1, y1))
    
    return result_img



def black_blur_fn(image_region):
    return Image.new("RGB", image_region.size, color=(0, 0, 0))

def pixelation_blur_fn(image_region, pixel_size=10):
    """
    Applies a pixelation effect by resizing down and up again.
    """
    # Get original size
    width, height = image_region.size

    # Resize down to small size (pixelate), then resize back up
    small = image_region.resize(
        (max(1, width // pixel_size), max(1, height // pixel_size)),
        resample=Image.NEAREST
    )
    pixelated = small.resize((width, height), Image.NEAREST)

    return pixelated