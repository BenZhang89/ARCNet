# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk


# Author: Ben Zhang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
from __future__ import absolute_import, print_function
import torch.nn as nn
import torch
import math
import pdb
import torch.nn.functional as F
import numpy as np



class ARCNet(nn.Module):
    """
    Inplementation of WNet, TNet and ENet presented in:
        Wang, Guotai, Wenqi Li, Sebastien Ourselin, and Tom Vercauteren. "Automatic brain tumor segmentation using cascaded anisotropic convolutional neural networks." arXiv preprint arXiv:1709.00382 (2017).
    These three variants are implemented in a single class named as "ARCNet".
    """
    def __init__(self,
                 num_classes,
                 vae_enable,
                 config,
                 start_channels=8,
                 in_channels=4,
                 acti_func='prelu',
                 name='ARCNet'):

        super(ARCNet, self).__init__()
        self.vae_enable = vae_enable
        self.acti_func = acti_func
        self.name = name
        self.config = config

        self.start_channels = start_channels #8
        self.down_channels_1 = 2 * self.start_channels 
        self.down_channels_2 = 2 * self.down_channels_1
        self.down_channels_3 = 2 * self.down_channels_2 
        
        # Encoder Blocks
        # Initial input has 4 channels, i.e t1,t1ce, t2, flair
        self.init_block = IniBlock(in_channels=in_channels, out_channels=self.start_channels)
        self.drop = nn.Dropout3d(0.2)
        self.block1 = ResBlock(in_channels=self.start_channels)
        self.down_1 = DownBlock(in_channels=self.start_channels, out_channels=self.down_channels_1)

        self.block_2_1 = ResBlock(in_channels=self.down_channels_1)
        self.block_2_2 = ResBlock(in_channels=self.down_channels_1)

        self.down_2 = DownBlock(in_channels=self.down_channels_1, out_channels=self.down_channels_2)

        self.block_3_1 = ResBlock(in_channels=self.down_channels_2)
        self.block_3_2 = ResBlock(in_channels=self.down_channels_2)

        self.down_3 = DownBlock(in_channels=self.down_channels_2, out_channels=self.down_channels_3)

        self.block_4_1 = ResBlock(in_channels=self.down_channels_3)
        self.block_4_2 = ResBlock(in_channels=self.down_channels_3)
        self.block_4_3 = ResBlock(in_channels=self.down_channels_3)
        self.block_4_4 = ResBlock(in_channels=self.down_channels_3)

        out_up_1_channels = int(self.down_channels_3 / 2) #64
        out_up_2_channels = int(out_up_1_channels / 2)
        out_up_3_channels = int(out_up_2_channels / 2) #8

        self.up_1 = UpBlock2(in_channels=self.down_channels_3, out_channels=out_up_1_channels)
        self.up_res_1 = ResBlock(in_channels=out_up_1_channels)

        self.up_2 = UpBlock2(in_channels=out_up_1_channels, out_channels=out_up_2_channels)
        self.up_res_2 = ResBlock(in_channels=out_up_2_channels)

        self.up_3 = UpBlock2(in_channels=out_up_2_channels, out_channels=out_up_3_channels)
        self.up_res_3 = ResBlock(in_channels=out_up_3_channels)

        self.end_block = IniBlock(in_channels=out_up_3_channels, out_channels=num_classes)
        # self.final_pred = OutputTransition(self.num_classes*(4+2+1), self.num_classes)
        # Variational Auto-Encoder
        if self.vae_enable:
            self.dense_features = (self.config['data_shape'][0]//3, self.config['data_shape'][1]//8, self.config['data_shape'][2]//8)
            self.vae = VAE(32, outChans=self.config['data_shape'][3], acti_func='prelu', dense_features=self.dense_features)

    def forward(self,x):
        x = self.init_block(x)
        #print(f'{x.shape}')
        x = self.drop(x)
        #print(f'{x.shape}')
        x1 = self.block1(x)
        #print(f'x1 {x1.shape}')
        x = self.down_1(x1)
        #print(f'{x.shape}')

        x = self.block_2_1(x)
        #print(f'{x.shape}')
        x2 = self.block_2_2(x)
        #print(f'x2 {x2.shape}')
        x = self.down_2(x2)
        #print(f'{x.shape}')

        x = self.block_3_1(x)
        #print(f'{x.shape}')
        x3 = self.block_3_2(x)
        #print(f'x3 {x3.shape}')
        x = self.down_3(x3)
        #print(f'{x.shape}')

        x = self.block_4_1(x)
        #print(f'{x.shape}')
        x = self.block_4_2(x)
        #print(f'{x.shape}')
        x = self.block_4_3(x)
        #print(f'{x.shape}')
        x4 = self.block_4_4(x)
        #print(f'x4 {x4.shape}')

        x = self.up_1(x4)
        #print(f'{x.shape}')
        x = self.up_res_1(x + x3)
        #print(f'{x.shape}')
        x = self.up_2(x)
        #print(f'{x.shape}')
        x = self.up_res_2(x + x2)
        #print(f'{x.shape}')
        x = self.up_3(x)
        #print(f'{x.shape}')
        x = self.up_res_3(x + x1)
        #print(f'{x.shape}')
        y = self.end_block(x)
        #print(y.shape)

        if self.vae_enable: 
            out_vae, out_distr = self.vae(f3)
            out_final = torch.cat((pred, out_vae), 1)
            return out_final, out_distr

        return y

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with '*.model'.
        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def predict(self, X, device=0, enable_dropout=False, out_prob=False):
        """
        Predicts the outout after the model is trained.
        Inputs:
        - X: Volume to be predicted
        """
        self.eval()
        if type(X) is np.ndarray:
            X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor).cuda(device, non_blocking=True)
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)

        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out = self.forward(X)

        if out_prob:
            return out
        else:
            max_val, idx = torch.max(out, 1)
            idx = idx.data.cpu().numpy()
            prediction = np.squeeze(idx)
            del X, out, idx, max_val
            return prediction


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels=32, norm="group"):
        super(ResBlock, self).__init__()
        if norm == "batch":
            norm_1 = nn.BatchNorm3d(num_features=in_channels)
            norm_2 = nn.BatchNorm3d(num_features=in_channels)
        elif norm == "group":
            norm_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
            norm_2 = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        self.layer_1 = nn.Sequential(
            norm_1,
            nn.ReLU())

        self.layer_2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            norm_2,
            nn.ReLU())

        self.conv_3 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3),
                                stride=1, padding=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        y = self.conv_3(x)
        y = y + x
        return y

class IniBlock(nn.Module):

    def __init__(self, in_channels, out_channels=8):
        super(IniBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                              stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                              stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UpBlock1(nn.Module):
    """
    TODO fix transpose conv to double spatial dim
    """

    def __init__(self, in_channels, out_channels):
        super(UpBlock1, self).__init__()
        self.transp_conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                                              stride=2, padding=1)

    def forward(self, x):
        return self.transp_conv(x)

class UpBlock2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpBlock2, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                                stride=1)
        # self.up_sample_1 = nn.Upsample(scale_factor=2, mode="bilinear") # TODO currently not supported in PyTorch 1.4 :(
        self.up_sample_1 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.up_sample_1(self.conv_1(x))
