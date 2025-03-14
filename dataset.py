import os 
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import re

def align_crop_opencv(img,
                      src_landmarks,
                      standard_landmarks,
                      crop_size=512,
                      face_factor=0.7,
                      align_type='similarity',
                      order=3,
                      mode='edge'):
    """Align and crop a face image by landmarks.

    Arguments:
        img                : Face image to be aligned and cropped.
        src_landmarks      : [[x_1, y_1], ..., [x_n, y_n]].
        standard_landmarks : Standard shape, should be normalized.
        crop_size          : Output image size, should be 1. int for (crop_size, crop_size)
                             or 2. (int, int) for (crop_size_h, crop_size_w).
        face_factor        : The factor of face area relative to the output image.
        align_type         : 'similarity' or 'affine'.
        order              : The order of interpolation. The order has to be in the range 0-5:
                                 - 0: INTER_NEAREST
                                 - 1: INTER_LINEAR
                                 - 2: INTER_AREA
                                 - 3: INTER_CUBIC
                                 - 4: INTER_LANCZOS4
                                 - 5: INTER_LANCZOS4
        mode               : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                             Points outside the boundaries of the input are filled according
                             to the given mode.
    """
    # set OpenCV
    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # crop size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size_h = crop_size[0]
        crop_size_w = crop_size[1]
    elif isinstance(crop_size, int):
        crop_size_h = crop_size_w = crop_size
    else:
        raise Exception('Invalid `crop_size`! `crop_size` should be 1. int for (crop_size, crop_size) or 2. (int, int) for (crop_size_h, crop_size_w)!')

    # estimate transform matrix
    trg_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array([crop_size_w // 2, crop_size_h // 2])
    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.inf)[0]

    # warp image by given transform
    output_shape = (crop_size_h, crop_size_w)
    img_crop = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order], borderMode=border[mode])

    # get transformed landmarks
    tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]

    return img_crop, tformed_landmarks


class CelebADataset(Dataset):
    def __init__(self, img_dir, transform=None, faceTransform=None, version="img_align_png", dims=None):
            # self.img_dir = os.path.abspath(os.path.join(img_dir, "img_align_celeba_png/"))

        self.version = version
        self.dims = dims
        if version == "img":
            # self.img_dir = os.path.join(img_dir, "img_celeba/")
            self.img_dir = os.path.abspath(os.path.join(img_dir, "img_celeba/"))
        elif version == "img_align": 
            # self.img_dir = os.path.join(img_dir, "img_align_celeba/")
            self.img_dir = os.path.abspath(os.path.join(img_dir, "img_align_celeba/"))
        elif version == "img_align_png": 
            # self.img_dir = os.path.join(img_dir, "img_align_celeba_png/")
            self.img_dir = os.path.abspath(os.path.join(img_dir, "img_align_celeba_png/"))
        else: 
            print("Invalid version chosen. Defaulting to smallest.")
            # self.img_dir = os.path.join(img_dir, "img_align_celeba_png/")
            self.img_dir = os.path.abspath(os.path.join(img_dir, "img_align_celeba_png/"))

        self.transform = transform
        self.image_filenames = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]
        self.faceTransform = faceTransform

        with open(os.path.join(self.img_dir, "../../Anno/list_bbox_celeba.txt"), "r") as file:
            lines = file.readlines()
            self.bbox_anno = {line.split()[0]: list(map(int, line.split()[1:])) for line in lines[2:]}

        with open(os.path.join(self.img_dir, "../../Anno/landmark.txt"), "r") as file:
            line = file.readline()
            n_landmark = len(re.split('[ ]+', line)[1:]) // 2

        self.landmark_anno = np.genfromtxt(os.path.join(self.img_dir, "../../Anno/landmark.txt"), dtype=float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1, n_landmark, 2)

        self.std_landmark_anno = np.genfromtxt(os.path.join(self.img_dir, "../../Anno/standard_landmark_68pts.txt"), dtype=float).reshape(n_landmark, 2)
        # self.landmark_anno = {
        #     row[0]: row[1:].reshape(n_landmark, 2) 
        #     for row in np.genfromtxt(os.path.join(self.img_dir, "../../Anno/landmark.txt"), dtype=float, usecols=range(n_landmark * 2 + 1))
        # }

        # self.std_landmark_anno = {
        #     row[0]: row[1:].reshape(n_landmark, 2) + np.array([0.25, 0.0])
        #     for row in np.genfromtxt(os.path.join(self.img_dir, "../../Anno/standard_landmark_68pts.txt"), dtype=float, usecols=range(n_landmark * 2 + 1))
        # }






        self.std_landmark_anno[:, 0] += 0.25 
        self.std_landmark_anno[:, 1] += 0.0


        with open(os.path.join(self.img_dir, "../../Anno/list_attr_celeba.txt"), "r") as file:
            lines = file.readlines()
            self.attribute_anno = {line.split()[0]: list(map(float, line.split()[1:])) for line in lines[2:]}

        with open(os.path.join(self.img_dir, "../../Eval/list_eval_partition.txt"), "r") as file:
            self.eval = {line.split()[0]: int(line.split()[1]) for line in file}


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image)

        if self.faceTransform:
            # Get face bounding box
            if self.version == "img":
                x, y, w, h = self.bbox_anno[filename]
                preImg = image[:, y:y+h, x:x+w]  # Keep all 3 channels (C, H, W)
                postImg = self.faceTransform(preImg)

                assert preImg.shape == postImg.shape, f"Shape mismatch: {preImg.shape} vs {postImg.shape}"

                image[:, y:y+h, x:x+w] = postImg


                # crop_size = image.shape[1:] if not self.dims else self.dims
                # print("cv")
                # image, _ = align_crop_opencv(np.array(image.permute(1, 2, 0) * 255, dtype=np.uint8),  # Convert tensor to OpenCV format
                #                              self.landmark_anno[int(filename[:-4])],  # Get source landmarks
                #                              self.std_landmark_anno,
                #                              crop_size=crop_size)
                
                # print("cv2")
                # image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

                
            elif self.version == "img_align" or self.version == "img_aling_png":
                pass

        return image

