import cv2
import torch
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np

from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.matlab_functions import imresize


@DATASET_REGISTRY.register()
class SingleImage_GT_Dataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImage_GT_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']
        self.cond_norm = opt['cond_norm']
        self.out_size = opt['out_size']
        self.downsample_list = opt['downsample_list']
        #degrads 
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.degrad_prob = opt['probab']
        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    def __getitem__(self, index):
        degrad_flag = False
        if np.random.uniform() < self.degrad_prob:
            degrad_flag = True
        # load lq image
        lq_path = self.paths[index]
        img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_gt = cv2.imread(lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        if degrad_flag:
            kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob)
            img_lq = cv2.filter2D(img_gt, -1, kernel)

        scale_ind = np.random.randint(len(self.downsample_list))
        scale = self.downsample_list[scale_ind]
        img_lq = imresize(img_lq, 1/scale)
        if degrad_flag:
            # noise
            img_lq = degradations.random_add_gaussian_noise(img_lq)
            # jpeg compression
            img_lq = degradations.random_add_jpg_compression(img_lq)
        img_lq = imresize(img_lq, scale)

        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)

        # BGR to RGB, HWC to CHW, numpy to tensor
        #img_lq = img2tensor(img_lq, bgr2rgb=False, float32=True)
        #img_gt = img2tensor(img_gt, bgr2rgb=False, float32=True)
        img_lq = torch.from_numpy(np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_gt = torch.from_numpy(np.transpose(img_gt[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        img_gt = torch.clamp((img_gt * 255.0).round(), 0, 255) / 255.
        in_size = scale / self.cond_norm
        cond = torch.from_numpy(np.array([in_size], dtype=np.float32)) 
        return {'lq': img_lq, 'lq_path': lq_path, 'in_size': cond, 'gt': img_gt}

    def __len__(self):
        return len(self.paths)
