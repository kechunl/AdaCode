import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import random
import pickle

sys.path.append('/newDisk/users/liukechun/research/FeMaSR')
from basicsr.utils import scandir


def main():
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for DIV2K, DIV8K, Flickr dataset.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.

        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each dataset, run this script.
        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3
    opt['label_dict'] = './scripts/data_preparation/label_dict.pkl'

    opt['input_folder'] = '/newDisk/dataset/DIV8K/train'
    opt['save_folder'] = '/newDisk/dataset/DIV8K/train_HR_sub'
    opt['semantic_folder'] = '/newDisk/users/liukechun/research/semantic-segmentation/output/DIV8K_test'
    opt['crop_size'] = 512 
    opt['step'] = 512 
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # FFHQ
    # opt['input_folder'] = '/newDisk/dataset/ffhq/val'
    # opt['save_folder'] = '/newDisk/dataset/ffhq/val_sub'
    # opt['crop_size'] = 512 
    # opt['maximum_num'] = 500
    # extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    # else:
    #     print(f'Folder {save_folder} already exists. Exit.')
    #     sys.exit(1)

    with open(opt['label_dict'], 'rb') as f:
        label_dict = pickle.load(f)

    img_list = list(scandir(input_folder, recursive=True, full_path=True))

    if opt.get('maximum_num') is not None:
        img_list = random.sample(img_list, opt['maximum_num'])

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    if 'ffhq' not in input_folder:
        patch_label = []
        for path in img_list:
            patch_label.append(pool.apply_async(worker, args=(path, opt, label_dict), callback=lambda arg: pbar.update(1)).get())
        patch_label = {k: v for d in patch_label for k, v in d.items()}
        with open(os.path.join(opt['save_folder'], 'label.pkl'), 'wb') as f:
            pickle.dump(patch_label, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        for path in img_list:
            pool.apply_async(worker_ffhq, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker_ffhq(path, opt):
    """Worker for each process for FFHQ dataset.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    img_name, extension = osp.splitext(osp.basename(path))

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    resize_scale = random.uniform(0.5, 1.0)
    img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)
    x = random.randint(0, img.shape[0] - crop_size)
    y = random.randint(0, img.shape[1] - crop_size)
    cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
    cropped_img = np.ascontiguousarray(cropped_img)
    cv2.imwrite(
        osp.join(opt['save_folder'], f'{img_name}{extension}'), cropped_img,
        [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    # print(process_info)
    return process_info


def worker(path, opt, label_dict):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # load label map
    labelmap = np.load(os.path.join(opt['semantic_folder'], os.path.basename(path).split('.')[0]+'.npy')).squeeze()
    # convert to grouped label
    labelmap = np.vectorize(label_dict.get)(labelmap)
    # scale to adjust labelmap
    scale = img.shape[0] / labelmap.shape[0]

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    patch_label = {}
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            label = assign_label(labelmap, int(x/scale), int(y/scale), int(crop_size/scale))
            patch_label[f'{img_name}_s{index:03d}{extension}'] = label
            # cv2.imwrite(
            #     osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
            #     [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    # process_info = f'Processing {img_name} ...'
    # return process_info
    return patch_label


def assign_label(labelmap, x, y, crop_size):
    labelcrop = labelmap[x:x+crop_size, y:y+crop_size]
    label = np.bincount(labelcrop.flatten()).argmax()
    return label


if __name__ == '__main__':
    main()
