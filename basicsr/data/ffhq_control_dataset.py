import cv2
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.matlab_functions import imresize


@DATASET_REGISTRY.register()
class FFHQ_control_Dataset(data.Dataset):

    def __init__(self, opt):
        super(FFHQ_control_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.degrad_prob = opt['probab']

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = paths_from_folder(self.gt_folder)

        # degradations
        self.downsample_list = opt['downsample_list']
        self.cond_norm = opt['cond_norm']

        logger = get_root_logger()

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img
    
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        degrad_flag = False
        if np.random.uniform() < self.degrad_prob:
            degrad_flag = True
        # load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape
        # ------------------------ generate lq image ------------------------ #
        if degrad_flag:
            kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob)
            img_lq = cv2.filter2D(img_gt, -1, kernel)
        
        scale_ind = np.random.randint(len(self.downsample_list))
        scale = self.downsample_list[scale_ind]
        img_lq = imresize(img_gt, 1/scale)
        if degrad_flag:
            # noise
            img_lq = degradations.random_add_gaussian_noise(img_lq)
            # jpeg compression
            img_lq = degradations.random_add_jpg_compression(img_lq)
        img_lq = imresize(img_lq, scale)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        
        in_size = scale / self.cond_norm
        cond = torch.from_numpy(np.array([in_size], dtype=np.float32)) 
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
       
        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'in_size': cond}

    def __len__(self):
        return len(self.paths)
