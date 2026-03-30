#! /usr/bin/python3

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
import h5py
import scipy.io

# code adopted from https://github.com/soyers/ddff-pytorch/blob/master/python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py


class DDFF12Loader_Val(Dataset):

    def __init__(self, args, mode):
        """
        Args:
            root_dir_fs (string): Directory with all focal stacks of all image datasets.
            root_dir_depth (string): Directory with all depth images of all image datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Disable opencv threading since it leads to deadlocks in PyTorch DataLoader
        hdf5_filename = args.hdf5_filename
        stack_key = args.stack_key
        disp_key = args.disp_key
        n_stack = args.n_stack
        min_disp = args.min_disp
        max_disp = args.max_disp

        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        self.max_n_stack = 10
        self.ignore_train = True

        assert n_stack <= self.max_n_stack, 'DDFF12 has maximum 10 images per stack!'
        self.n_stack = n_stack
        assert self.n_stack <= self.max_n_stack, "For validation, we can never use more than 10 images since the original DDFF12 stacks only have 10 images"
        self.disp_dist = torch.linspace(max_disp,min_disp, steps=self.max_n_stack)

        assert 'val' in self.stack_key, "stack_key should contain 'val' since this is the validation loader. Got stack_key: {self.stack_key}"

        transform_test = [DDFF12Loader_Val.ToTensor(),
                            DDFF12Loader_Val.PadSamples((384, 576)),
                            DDFF12Loader_Val.Normalize(mean_input=[0.485, 0.456, 0.406],
                                                    std_input=[0.229, 0.224, 0.225])]
        self.transform =  torchvision.transforms.Compose(transform_test)


        # Load intrinsics matrix
        calib_mat_file_path = "dataset/datasets/ddff12_val_generation/third_part/IntParamLF.mat"
        assert os.path.exists(calib_mat_file_path), f"DDFF12 intrinsics file not found at {calib_mat_file_path}!'"
        mat = scipy.io.loadmat(calib_mat_file_path)
        mat = np.squeeze(mat['IntParamLF'])
        self.K2 = mat[1]
        if self.K2 >1983 or self.K2 < 1982:
            raise ValueError("DDFF12 intrinsics K2 value seems off, expected around 1982-1983.")
        fxy = mat[2:4]
        self.flens = max(fxy)
        self.fsubaperture = 521.4052 # pixel
        self.baseline = self.K2/self.flens*1e-3 # meters

        # From page 7 of Hazirbas et al. -- https://arxiv.org/pdf/1704.01085
        K = np.array([[521.4, 0, 285.11],
                        [0, 521.4, 187.83],
                        [0, 0, 1]])
        self.K = torch.Tensor(K)

    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        # Create sample dict

        # Returns: 
        try:
            if 'test' in self.stack_key:
                sample =  {'input': self.hdf5[self.stack_key][idx].astype(float), 'output': np.ones([2,2])}
            else:
                sample = {'input': self.hdf5[self.stack_key][idx].astype(float), 'output': self.hdf5[self.disp_key][idx]}
        except:
            sample = None
            for _ in range(100):
                sample = {'input': self.hdf5[self.stack_key][idx].astype(float), 'output': self.hdf5[self.disp_key][idx]}
                if sample is not None:
                    break
            if sample is None:
                a =  self.hdf5[self.stack_key][idx].astype(float)
                b = self.hdf5[self.disp_key][idx]
                print(len(self.hdf5[self.stack_key]), idx, a is None, b is None)
                exit(1)
        # Transform sample with data augmentation transformers
        if self.transform :
            sample_out = self.transform(sample)

        # This seems unintentional in the original dataloader, but it samples [0,2,4,6,9] when n_stack=5 since [0, 2.25, 4.5,  6.75, 9] rounds down
        rand_idx = np.linspace(0, 9, self.n_stack)  

        out_imgs = sample_out['input'][rand_idx]
        out_disp = sample_out['output']
        disp_dist = self.disp_dist[rand_idx]
        
        
        dep = (self.baseline*self.fsubaperture)/out_disp # in meters
        fd_list = (self.baseline*self.fsubaperture)/disp_dist # in meters

        depth_valid_mask = (out_disp > 0.0).cpu().numpy().squeeze()

        # Pick a dummy as rgb
        output = {'rgb': out_imgs[0], 'gt': dep, 'K': self.K, 'valid_mask': depth_valid_mask, 'fd_list': fd_list, 'focal_stack': out_imgs}

        

        return output #sample_out['input'], sample_out['output'].squeeze()#, sample['input']

    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""

        def __call__(self, sample):
            # Add color dimension to depth map
            sample['output'] = np.expand_dims(sample['output'], axis=0)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            sample['input'] = torch.from_numpy(sample['input'].transpose((0, 3, 1, 2))).float().div(255) #I add div 255
            sample['output'] = torch.from_numpy(sample['output']).float()
            return sample

    class Normalize(object):
        def __init__(self, mean_input, std_input, mean_output=None, std_output=None):
            self.mean_input = mean_input
            self.std_input = std_input
            self.mean_output = mean_output
            self.std_output = std_output

        def __call__(self, sample):
            img_lst = []
            samples = sample['input']

            for i, sample_input in enumerate(samples):
                img_lst.append(torchvision.transforms.functional.normalize(sample_input, mean=self.mean_input, std=self.std_input))
            input_images = torch.stack(img_lst)

            if self.mean_output is None or self.std_output is None:
                output_image = sample['output']
            else:
                output_image = torchvision.transforms.functional.normalize(sample['output'], mean=self.mean_output,
                                                                    std=self.std_output)


            return {'input': input_images, 'output': output_image}

    class PadSamples(object):
        def __init__(self, output_size, ground_truth_pad_value=0.0):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
            self.ground_truth_pad_value = ground_truth_pad_value

        def __call__(self, sample):
            h, w = sample['input'].shape[2:4]
            new_h, new_w = self.output_size
            padh = np.int32(new_h - h)
            padw = np.int32(new_w - w)
            sample['input'] = torch.stack(
                [torch.from_numpy(np.pad(sample_input.numpy(), ((0, 0), (0, padh), (0, padw)), mode="reflect")).float()
                 for sample_input in sample['input']])
            sample['output'] = torch.from_numpy(
                np.pad(sample['output'].numpy(), ((0, 0), (0, padh), (0, padw)), mode="constant",
                       constant_values=self.ground_truth_pad_value)).float()

            return sample