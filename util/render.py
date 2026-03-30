# MIT License

# Copyright (c) 2018 shirgur

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Modified from Single Image Depth Estimation Trained via Depth from Defocus Cues
# (https://github.com/shirgur/UnsupervisedDepthFromFocus) by Shir Gur et al.

import torch 
import torch.nn as nn
import power_exp_psf_cuda as power_exp_psf

class PowerExpPSF(nn.Module):
    def __init__(self, kernel_size, p):
        super(PowerExpPSF, self).__init__()
        self.kernel_size = kernel_size
        self.p = p

    def forward(self, image, psf):
        psf = psf.expand_as(image).contiguous()
        x = torch.arange(self.kernel_size // 2,
                            -self.kernel_size // 2,
                            -1).view(self.kernel_size, 1).float().repeat(1, self.kernel_size).to(image.device)

        y = torch.arange(self.kernel_size // 2,
                            -self.kernel_size // 2,
                            -1).view(1, self.kernel_size).float().repeat(self.kernel_size, 1).to(image.device)
        outputs, _ = power_exp_psf.forward(image, self.p, psf, x, y)
        return outputs