from retinaface import RetinaFace
import cv2
from PIL import Image, ImageFilter
import numpy as np


def blur_face(image, blur_sigma):
    # convert PIL image to cv so retinaface can do its thing
    img_np = np.array(image)
    img_cv = img_np[:, :, ::-1].copy()  # RGB to BGR

    faces = RetinaFace.detect_faces(img_cv)

    # if no faces then send it back lol
    if faces is None or len(faces) == 0:
        return image

    result_img = image.copy()

    # for each face (should be only 1 'cause that'd be weird)
    for i in faces:
        face = faces[i]
        facial_area = face['facial_area']

        x1, y1, x2, y2 = facial_area

        # crop and blur it boss
        face_region = result_img.crop((x1, y1, x2, y2))

        blurred_face = face_region.filter(ImageFilter.GaussianBlur(blur_sigma))

        # put it back
        result_img.paste(blurred_face, (x1, y1))

    return result_img