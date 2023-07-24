import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool
import argparse

from basicsr.data.bsrgan_util import degradation_bsrgan

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf"), followlinks=True):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=followlinks)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def degrade_img(hr_path, save_path):
    img_gt = cv2.imread(hr_path).astype(np.float32) / 255.
    img_gt = img_gt[:, :, [2, 1, 0]] # BGR to RGB
    img_lq, img_gt = degradation_bsrgan(img_gt, sf=scale, use_crop=False)
    img_lq = (img_lq[:, :, [2, 1, 0]] * 255).astype(np.uint8)
    print(f'Save {save_path}')
    cv2.imwrite(save_path, img_lq)

argparser = argparse.ArgumentParser(description='Generate degradation dataset following BSRGAN')
argparser.add_argument('--scale', '-s', type=int, choices=[2, 4], help='downscale factor')
argparser.add_argument('--dir', type=str, help='image directory')
args = argparser.parse_args()

seed = 123
random.seed(seed)
np.random.seed(seed)

scale = args.scale
hr_img_list = make_dataset(args.dir)
pool = Pool(processes=40)

for hr_path in hr_img_list:
    try:
        save_dir = os.path.dirname(hr_path) + f'_LR_X{scale}'
        save_path = os.path.join(save_dir, os.path.basename(hr_path))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        pool.apply_async(degrade_img(hr_path, save_path))
    except:
        print(hr_path, ': LR not generated.')

pool.close()
pool.join()

