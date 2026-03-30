# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
import numpy as np
import gc
import sys
import os
import math

from .backbone import FST
from .dpt import DPTHead

class FOSSA(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        last_layer="sigmoid",
        max_depth=1.0,
        temporal_fuse_method=None,
        num_layers_until_collapse=6,
        fd_embed_function='none',
        turn_off_motion_module=False,
    ):
        super(FOSSA, self).__init__()
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.encoder = encoder

        self.pretrained = FST(model_name=encoder, num_layers_until_collapse=num_layers_until_collapse, temporal_fuse_method=temporal_fuse_method, num_frames=num_frames, pe=pe, turn_off_motion_module=turn_off_motion_module)
        self.fd_embed_function = fd_embed_function
        assert fd_embed_function in ['none', 'inverse', 'log_plus_1'], f"fd_embed_function must be one of ['none', 'inverse', 'log_plus_1'], got {fd_embed_function}"

        # self.depth_head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, last_layer=last_layer, pe=pe)
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, last_layer=last_layer)
        self.max_depth = max_depth
        assert temporal_fuse_method is not None, "temporal_fuse_method must be specified for FOSSA model"
        self.temporal_fuse_method = temporal_fuse_method
    def forward(self, x, fd_list):
        """
        x: (B, N, 3, H, W) focal stack input
        fd_list: (B, N) focal distance list
        return: (B, N, H, W) depth map for each focal plane
        """
        # Could do this to get contiguous memory format to get convolution operation to be the same as DAv2
        # x = x.contiguous(memory_format=torch.contiguous_format)
        original_batch_size, num_focal_planes, channels, original_height, original_width = x.shape

        x = x.view(original_batch_size * num_focal_planes, channels, original_height, original_width)


        x = self.resize_preserve_aspect(x, ensure_multiple_of=14)


        _, _, new_height, new_width = x.shape

        x = x.view(original_batch_size, num_focal_planes, channels, new_height, new_width)


        B, T, C, H, W = x.shape

        patch_h, patch_w = H // 14, W // 14

        if self.fd_embed_function == 'inverse':
            fd_used = fd_list.clone().reciprocal()
        elif self.fd_embed_function == 'log_plus_1':
            fd_used = torch.log10(fd_list) + 1.0
        elif self.fd_embed_function == 'none':
            fd_used = fd_list
        else:
            raise ValueError(f"fd_embed_function must be one of ['none', 'inverse', 'log_plus_1'], got {self.fd_embed_function}")

        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True, fd_list=fd_used, actual_batch_size=original_batch_size, frame_length=T, temporal_fuse_method=self.temporal_fuse_method)

        depth = self.depth_head(out_features=features, patch_h =patch_h, patch_w=patch_w) * self.max_depth


        depth = F.interpolate(
            depth,
            size=(original_height, original_width),  # Original H, W
            mode='bilinear',
            align_corners=True
        )

        depth = F.relu(depth) # This is not necessary if the final layer of the depth_head is a non-negative layer like ReLU or Sigmoid or Softplus

        return depth


    def resize_preserve_aspect(self, input_image, short_edge_target_length=518, ensure_multiple_of=14):
        h = input_image.shape[-2]
        w = input_image.shape[-1]

        if h < w:
            new_h = short_edge_target_length
            new_w = int(w * (short_edge_target_length / h))
        else:
            new_w = short_edge_target_length
            new_h = int(h * (short_edge_target_length / w))

        assert ensure_multiple_of >= 1, f"ensure_multiple_of must be greater than or equal to 1, got {ensure_multiple_of}"


        new_height = math.ceil(new_h / ensure_multiple_of) * ensure_multiple_of
        new_width = math.ceil(new_w / ensure_multiple_of) * ensure_multiple_of

        resized_image = F.interpolate(input_image, size=(new_height, new_width), mode='bilinear', align_corners=True)


        return resized_image
