import os
import re
import numpy as np
import tqdm
from functools import partial
from multiprocessing import Pool
import cropper  # Assuming cropper.py is imported

import cv2
imread = cv2.imread
imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), 95])
align_crop = cropper.align_crop_opencv

def process_images(img_dir,
                   save_dir,
                   landmark_file,
                   standard_landmark_file,
                   crop_size=(572, 572),
                   move=(0.25, 0.0),
                   save_format='jpg',
                   n_workers=8,
                   face_factor=0.45,
                   align_type='similarity',
                   order=3,
                   mode='edge'):
    """
    Process images by aligning and cropping faces using provided landmarks.

    Args:
        img_dir (str): Path to the image directory.
        save_dir (str): Path to save aligned images.
        landmark_file (str): Path to the landmark file.
        standard_landmark_file (str): Path to the standard landmarks file.
        crop_size (tuple): (Height, Width) of cropped images.
        move (tuple): (move_h, move_w), adjustments to standard landmarks.
        save_format (str): Output format ('jpg' or 'png').
        n_workers (int): Number of workers for multiprocessing.
        face_factor (float): Factor of face area relative to the output image.
        align_type (str): 'affine' or 'similarity' transformation.
        order (int): Interpolation order (0 to 5).
        mode (str): Border mode ('constant', 'edge', etc.).

    Returns:
        str: Path to the saved landmark file.
    """

    # Read landmarks
    with open(landmark_file) as f:
        line = f.readline()
    n_landmark = len(re.split('[ ]+', line)[1:]) // 2

    img_names = np.genfromtxt(landmark_file, dtype=str, usecols=0)
    landmarks = np.genfromtxt(landmark_file, dtype=float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1, n_landmark, 2)
    standard_landmark = np.genfromtxt(standard_landmark_file, dtype=float).reshape(n_landmark, 2)

    # Adjust standard landmarks based on move values
    standard_landmark[:, 0] += move[1]
    standard_landmark[:, 1] += move[0]

    # Create save directory
    save_dir = os.path.join(save_dir, f'align_size({crop_size[0]},{crop_size[1]})_move({move[0]:.3f},{move[1]:.3f})_face_factor({face_factor:.3f})_{save_format}')
    data_dir = os.path.join(save_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    def process_single_image(i):
        """Process a single image."""
        for _ in range(3):  # Try up to 3 times in case of failure
            try:
                img = imread(os.path.join(img_dir, img_names[i]))
                img_crop, tformed_landmarks = align_crop(
                    img,
                    landmarks[i],
                    standard_landmark,
                    crop_size=crop_size,
                    face_factor=face_factor,
                    align_type=align_type,
                    order=order,
                    mode=mode
                )

                # Save image
                name = os.path.splitext(img_names[i])[0] + '.' + save_format
                path = os.path.join(data_dir, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                imwrite(path, img_crop)

                # Save transformed landmarks
                tformed_landmarks.shape = -1
                return ('%s' + ' %.1f' * n_landmark * 2) % ((name,) + tuple(tformed_landmarks))

            except Exception as e:
                print(f"Error processing {img_names[i]}: {e}")
                continue
        return None

    # Use multiprocessing to process images faster
    with Pool(n_workers) as pool:
        name_landmark_strs = list(tqdm.tqdm(pool.imap(process_single_image, range(len(img_names))), total=len(img_names)))

    # Save new landmark file
    landmarks_path = os.path.join(save_dir, 'landmark.txt')
    with open(landmarks_path, 'w') as f:
        for name_landmark_str in name_landmark_strs:
            if name_landmark_str:
                f.write(name_landmark_str + '\n')

    print(f"Processing complete. Landmarks saved at {landmarks_path}")
    return landmarks_path

# process_images(
#     img_dir="./data/img_celeba",
#     save_dir="./data/aligned",
#     landmark_file="./data/landmark.txt",
#     standard_landmark_file="./data/standard_landmark_68pts.txt",
#     crop_size=(572, 572),
#     move=(0.25, 0.0),
#     save_format='jpg',
#     n_workers=8,
#     face_factor=0.45,
#     align_type='similarity',
#     order=3,
#     mode='edge'
# )
