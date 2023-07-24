import os
import cv2
import random
import numpy as np
from torch.utils import data as data
import math
from PIL import Image, ImageDraw

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from .data_util import make_dataset


def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def brush_stroke_mask(img, color=(255,255,255)):
    min_num_vertex = 8
    max_num_vertex = 28
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('RGB', (W, H), 0)
        if img is not None: mask = img #Image.fromarray(img)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=color, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=color)

        return mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.same_lq_size = opt.get('same_lq_size', False)
        
        if opt.get('dataroot_gt') is not None:
            self.gt_folder = opt['dataroot_gt']
            self.gt_paths = make_dataset(self.gt_folder)
        elif opt.get('datafile_gt') is not None:
            with open(opt.get('datafile_gt'), "r") as f:
                paths = f.read().splitlines()
            self.gt_paths = sorted(paths)
        else:
            raise ValueError("Unknown path for gt.")

        if opt.get('dataroot_lq') is not None:
            self.lq_folder = opt['dataroot_lq']
            self.lq_paths = make_dataset(self.lq_folder)
        elif opt.get('datafile_lq') is not None:
            with open(opt.get('datafile_lq'), "r") as f:
                paths = f.read().splitlines()
            self.lq_paths = sorted(paths)
        else:
            raise ValueError("Unknown path for lq.")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        #  scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.
        lq_path = self.lq_paths[index]
        img_lq = cv2.imread(lq_path).astype(np.float32) / 255.

        # augmentation for training
        if self.opt['phase'] == 'train':
            input_gt_size = img_gt.shape[0]
            input_lq_size = img_lq.shape[0]
            scale = input_gt_size // input_lq_size
            gt_size = self.opt['gt_size']

            if self.opt['use_resize_crop']:
                # random resize
                input_gt_random_size = random.randint(gt_size, input_gt_size)
                input_gt_random_size = input_gt_random_size - input_gt_random_size % scale # make sure divisible by scale 
                resize_factor = input_gt_random_size / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
                img_lq = random_resize(img_lq, resize_factor)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, input_gt_size // input_lq_size,
                                               gt_path)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        if self.opt['phase'] != 'train':
            crop_eval_size = self.opt.get('crop_eval_size', None)
            if crop_eval_size:
                input_gt_size = img_gt.shape[0]
                input_lq_size = img_lq.shape[0]
                scale = input_gt_size // input_lq_size
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, crop_eval_size, input_gt_size // input_lq_size,
                                               gt_path)

        if self.same_lq_size:
            img_lq = random_resize(img_lq, img_gt.shape[0] / img_lq.shape[0])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)

@DATASET_REGISTRY.register()
class InpaintingImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(InpaintingImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.same_lq_size = opt.get('same_lq_size', False)

        if opt.get('dataroot_gt') is not None:
            self.gt_folder = opt['dataroot_gt']
            self.gt_paths = make_dataset(self.gt_folder)
        elif opt.get('datafile_gt') is not None:
            with open(opt.get('datafile_gt'), "r") as f:
                paths = f.read().splitlines()
            self.gt_paths = sorted(paths)
        else:
            raise ValueError("Unknown path for gt.")

        # if opt.get('dataroot_lq') is not None:
        #     self.lq_folder = opt['dataroot_lq']
        #     self.lq_paths = make_dataset(self.lq_folder)
        # elif opt.get('datafile_lq') is not None:
        #     with open(opt.get('datafile_lq'), "r") as f:
        #         paths = f.read().splitlines()
        #     self.lq_paths = sorted(paths)
        # else:
        #     raise ValueError("Unknown path for lq.")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        #  scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_gt = cv2.imread(gt_path)
        img_lq = np.asarray(brush_stroke_mask(Image.fromarray(img_gt))).astype(np.float32) / 255.
        img_gt = img_gt.astype(np.float32) / 255.
        # lq_path = self.lq_paths[index]
        # img_lq = cv2.imread(lq_path).astype(np.float32) / 255.
        # augmentation for training
        if self.opt['phase'] == 'train':
            input_gt_size = img_gt.shape[0]
            input_lq_size = img_lq.shape[0]
            scale = input_gt_size // input_lq_size
            gt_size = self.opt['gt_size']

            if self.opt['use_resize_crop']:
                # random resize
                input_gt_random_size = random.randint(gt_size, input_gt_size)
                input_gt_random_size = input_gt_random_size - input_gt_random_size % scale  # make sure divisible by scale
                resize_factor = input_gt_random_size / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
                img_lq = random_resize(img_lq, resize_factor)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, input_gt_size // input_lq_size,
                                                    gt_path)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        if self.opt['phase'] != 'train':
            crop_eval_size = self.opt.get('crop_eval_size', None)
            if crop_eval_size:
                input_gt_size = img_gt.shape[0]
                input_lq_size = img_lq.shape[0]
                scale = input_gt_size // input_lq_size
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, crop_eval_size, input_gt_size // input_lq_size,
                                                    gt_path)

        if self.same_lq_size:
            img_lq = random_resize(img_lq, img_gt.shape[0] / img_lq.shape[0])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)

