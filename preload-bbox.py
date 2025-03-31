import torch
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.blurring_utils import detect_face, blur_face
from utils.lfw_utils import *
from utils.env_utils import random_crop_pil as random_crop

root_dir = 'data/lfw/'

# Set up paths
people_dir = os.path.join(root_dir, 'lfw-deepfunneled', 'lfw-deepfunneled')

# Get all image paths and labels
all_people = []
image_paths = [] # only bros with more than one pic
labels = []
names = []

person_folders = os.listdir(people_dir)

label_idx = 0
label_map = {}

for person in person_folders:
    person_dir = os.path.join(people_dir, person)
    if os.path.isdir(person_dir):
        person_images = os.listdir(person_dir)

        # we want bros with more than one pic
        if len(person_images) > 1:
            if person not in label_map:
                label_map[person] = label_idx
                label_idx += 1

            if person not in all_people:
                all_people.append(person)

            for img_name in person_images:
                if img_name.endswith('.jpg'):
                    image_paths.append(os.path.join(person_dir, img_name))
                    labels.append(label_map[person])
                    names.append(person)

import cv2 as cv
import pickle

result_bbox = {}

for image_path in image_paths:
    img = cv.imread(image_path)
    result_bbox[image_path] = detect_face(img)
    print(image_path, result_bbox[image_path])

with open('result_bbox.pickle', 'wb') as handle:
    pickle.dump(result_bbox, handle, protocol=pickle.HIGHEST_PROTOCOL)