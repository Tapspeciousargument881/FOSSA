import numpy as np
from importlib import import_module
from torch.utils.data import Dataset

import torch

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, Resize
import cv2

PATTERN_IDS = {
    'random': 0,
    'velodyne': 1,
    'sfm': 2
}

def get_center_crop_origin(image_size, crop_size):
    h_img, w_img = image_size
    crop_h, crop_w = crop_size
    i = (h_img - crop_h) // 2
    j = (w_img - crop_w) // 2
    return i, j

def get(args, mode):
    if mode == "train":
        data_name = args.train_data_name
    elif mode == "val" or mode == "test":
        data_name = args.val_data_name
    else:
        raise NotImplementedError

    data_names = data_name.split("+")
    if len(data_names) == 1: # use the original dataset
        module_name = 'dataset.' + data_name.lower() # Load in vkitti.py if --train_data_name VKITTI
        dataset_name = data_name
    else:
        module_name = 'dataset.multidataset'
        dataset_name = 'MultiDataset'

    module = import_module(module_name)
    return getattr(module, dataset_name)(args, mode=mode) # Returns the initialized dataset with the args and mode


class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode # training or validation
        self.max_depth_range = 100.0

    # Method that must be extended in the child class (like vkitti.py)
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

    # Common method for all datasets to preprocess RGB images, depth maps, and camera intrinsics
    def process(self, rgb, dep, K, depth_valid_mask, normalize_median=True, random_crop=True, ddff12_focus_stack=None):
        args = self.args
        assert self.mode == 'val', f"Currently only supports processing for validation samples, but got mode={self.mode}"
        assert self.augment == False, f"Currently only supports processing for non-augmented samples during evaluation, but got augment={self.augment}"

        t_rgb = T.Compose([
            T.Resize((self.resize_height, self.resize_width)),
            T.CenterCrop(self.crop_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_rgb_np_raw = T.Compose([
            T.Resize((self.resize_height, self.resize_width)),
            T.CenterCrop(self.crop_size),
            self.ToNumpy()
        ])

        t_dep = T.Compose([
            T.Resize((self.resize_height, self.resize_width), InterpolationMode.NEAREST),
            T.CenterCrop(self.crop_size),
            self.ToNumpy(),
            T.ToTensor()
        ])

        t_depth_mask = T.Compose([
            T.Resize((self.resize_height, self.resize_width), InterpolationMode.NEAREST),
            T.CenterCrop(self.crop_size)
        ])

        

        original_w, original_h = rgb.size
        scale_ratio_h, scale_ratio_w = self.resize_height / original_h, self.resize_width / original_w
        assert np.isclose(scale_ratio_h, scale_ratio_w, rtol=0.01), "only support resizing that keeps the original aspect ratio"

        i, j = get_center_crop_origin((self.resize_height, self.resize_width), self.crop_size)

        K = K.clone()

        # adjust focal for resizing
        K[0] = K[0] * scale_ratio_h
        K[1] = K[1] * scale_ratio_w

        # adjust principal point for cropping
        K[0, 2] -= j
        K[1, 2] -= i

        rgb_final = t_rgb(rgb)
        dep = t_dep(dep)
        rgb_np_raw = t_rgb_np_raw(rgb)
        depth_valid_mask = t_depth_mask(depth_valid_mask)

        # Normalize the depth map to be mean 1
        if normalize_median:
            # compute the median
            median = torch.median(dep[dep > 0.0])
            dep = dep / median # normalize so that median is 1

        # Replaces NaN values with 0
        dep = torch.nan_to_num(dep)
        # Convert depth_valid_mask to a np array
        depth_valid_mask = np.array(depth_valid_mask).astype(bool)
        
        return rgb_final, dep, K, depth_valid_mask, rgb_np_raw, ddff12_focus_stack

    def refresh_indices(self):
        pass 
