from insightface.app import FaceAnalysis
import cv2
from PIL import Image, ImageFilter
import numpy as np

def detect_face(image):
    # Initialize InsightFace FaceAnalysis
    face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    # Convert PIL image to cv2 format for InsightFace
    img_np = np.array(image)
    img_cv = img_np[:, :, ::-1].copy()  # RGB to BGR

    # Detect faces using InsightFace
    faces = face_analyzer.get(img_cv)

    return faces


def blur_face(image, blur_sigma):
    faces = detect_face(image)
    # If no faces then return the original image
    if faces is None or len(faces) == 0:
        return image

    result_img = image.copy()

    # For each detected face
    for face in faces:
        # Get the bounding box coordinates
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)

        # Only process if we have a valid region
        if x2 > x1 and y2 > y1:
            # Crop and blur the face region
            face_region = result_img.crop((x1, y1, x2, y2))
            blurred_face = face_region.filter(ImageFilter.GaussianBlur(blur_sigma))

            # Paste the blurred face back into the image
            result_img.paste(blurred_face, (x1, y1))

    return result_img