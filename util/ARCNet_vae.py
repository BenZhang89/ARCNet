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
                 acti_func='prelu',
                 name='ARCNet'):

        super(ARCNet, self).__init__()
        self.num_classes = num_classes
        self.vae_enable = vae_enable
        self.acti_func = acti_func
        self.base_chns = [32, 32, 32] #[32, 32, 32, 32]
        self.downsample_twice = True
        self.name = name
        self.config = config
        
        # Encoder Blocks
        # Initial input has 4 channels, i.e t1,t1ce, t2, flair
        self.in_conv0 = DownSampling(4, outChans=32, stride=1,dropout_rate=0.2)
        self.block1_1 = ResBlock(self.base_chns[0],self.base_chns[0],kernels = [(1, 3, 3), (1, 3, 3)],
            acti_func=self.acti_func,name='block1_1')
        self.block1_2 = ResBlock(self.base_chns[0],self.base_chns[0],kernels = [(1, 3, 3), (1, 3, 3)],
            acti_func=self.acti_func,name='block1_2')

        self.block2_1 = ResBlock(self.base_chns[0],self.base_chns[1],kernels = [(1, 3, 3), (1, 3, 3)],
            acti_func=self.acti_func,name='block2_1')
        self.block2_2 = ResBlock(self.base_chns[1],self.base_chns[1],kernels = [(1, 3, 3), (1, 3, 3)],
            acti_func=self.acti_func,name='block2_2')

        self.block3_1 = ResBlock(self.base_chns[1],self.base_chns[2],kernels = [(1, 3, 3), (1, 3, 3)],
            dilation_rates = [(1, 1, 1), (1, 1, 1)],acti_func=self.acti_func,name='block3_1')
        self.block3_2 = ResBlock(self.base_chns[2],self.base_chns[2],kernels = [(1, 3, 3), (1, 3, 3)],
            dilation_rates = [(1, 2, 2), (1, 2, 2)],paddings=[(0,2,2),(0,2,2)],acti_func=self.acti_func,name='block3_2')
        self.block3_3 = ResBlock(self.base_chns[2],self.base_chns[2],kernels = [(1, 3, 3), (1, 3, 3)],
            dilation_rates = [(1, 3, 3), (1, 3, 3)],paddings=[(0,3,3),(0,3,3)],acti_func=self.acti_func,name='block3_3')

        self.block4_1 = ResBlock(self.base_chns[2],self.base_chns[2],kernels = [(1, 3, 3), (1, 3, 3)],
            dilation_rates = [(1, 3, 3), (1, 3, 3)],paddings=[(0,3,3),(0,3,3)],acti_func=self.acti_func,name='block4_1')
        self.block4_2 = ResBlock(self.base_chns[2],self.base_chns[2],kernels = [(1, 3, 3), (1, 3, 3)],
            dilation_rates = [(1, 2, 2), (1, 2, 2)],paddings=[(0,2,2),(0,2,2)],acti_func=self.acti_func,name='block4_2')
        self.block4_3 = ResBlock(self.base_chns[2],self.base_chns[2],kernels = [(1, 3, 3), (1, 3, 3)],
            dilation_rates = [(1, 1, 1), (1, 1, 1)],acti_func=self.acti_func,name='block4_3')


        self.fuse1 = ConvolutionalLayer(self.base_chns[0], self.base_chns[0],
            kernel_size= (3, 1, 1),acti_func=self.acti_func,name='fuse1')
        self.downsample1 = ConvolutionalLayer(self.base_chns[0], self.base_chns[0],
            kernel_size= (1, 3, 3),stride = (1, 2, 2),padding=(0,1,1), acti_func=self.acti_func,name='downsample1')      

        self.fuse2 = ConvolutionalLayer(self.base_chns[1], self.base_chns[1],
            kernel_size= (3, 1, 1),acti_func=self.acti_func,name='fuse2')
        self.downsample2 = ConvolutionalLayer(self.base_chns[1], self.base_chns[1],
            kernel_size= (1, 3, 3),stride = (1, 2, 2),padding=(0,1,1),acti_func=self.acti_func,name='downsample2')  

        self.fuse3 = ConvolutionalLayer(self.base_chns[2], self.base_chns[2],
            kernel_size= (3, 1, 1),acti_func=self.acti_func,name='fuse3')

        self.fuse4 = ConvolutionalLayer(self.base_chns[2], self.base_chns[2],
            kernel_size= (3, 1, 1), acti_func=self.acti_func,name='fuse4')

        self.feature_expand1 = ConvolutionalLayer(self.base_chns[1], self.base_chns[1],
            kernel_size= (1, 1, 1), stride = (1, 1, 1), acti_func=self.acti_func, name='feature_expand1')

        self.feature_expand2 = ConvolutionalLayer(self.base_chns[2], self.base_chns[2],
            kernel_size= (1, 1, 1), stride = (1, 1, 1), acti_func=self.acti_func, name='feature_expand2')

        self.feature_expand3 = ConvolutionalLayer(self.base_chns[2], self.base_chns[2],
            kernel_size= (1, 1, 1), stride = (1, 1, 1), acti_func=self.acti_func, name='feature_expand3')

        self.centra_slice1 = TensorSliceLayer(margin = 2)
        self.centra_slice2 = TensorSliceLayer(margin = 1)
        self.pred1 = nn.Conv3d(in_channels=self.base_chns[2], out_channels=self.num_classes,
                        kernel_size=(1, 3, 3), bias=False)
        #different kernel size compared to cacnn
        self.pred_up1  = DeconvolutionalLayer(self.base_chns[2], self.num_classes, kernel_size= (1, 2, 2),
            stride = (1, 2, 2), padding=0, acti_func=self.acti_func, name='pred_up1')

        self.pred_up2_1  = DeconvolutionalLayer(self.base_chns[2], self.num_classes*2, kernel_size= (1, 2, 2),
            stride = (1, 2, 2), padding=0,acti_func=self.acti_func, name='pred_up2_1')
        self.pred_up2_2  = DeconvolutionalLayer(self.num_classes*2, self.num_classes*2, kernel_size= (1, 2, 2),
            stride = (1, 2, 2), padding=0,acti_func=self.acti_func, name='pred_up2_2')

        self.pred_up3_1  = DeconvolutionalLayer(self.base_chns[2], self.num_classes*4, kernel_size= (1, 2, 2),
            stride = (1, 2, 2), padding=0,acti_func=self.acti_func, name='pred_up3_1')
        self.pred_up3_2  = DeconvolutionalLayer(self.num_classes*4, self.num_classes*4, kernel_size= (1, 2, 2),
            stride = (1, 2, 2), padding=0,acti_func=self.acti_func, name='pred_up3_2')

        self.final_pred =  nn.Conv3d(in_channels=self.num_classes*(4+2+1), out_channels=self.num_classes,
                        kernel_size=(1, 3, 3), padding=(0,1,1), bias=False)
        # Variational Auto-Encoder
        if self.vae_enable:
            self.dense_features = (self.config['data_shape'][0]//3, self.config['data_shape'][1]//8, self.config['data_shape'][2]//8)
            self.vae = VAE(32, outChans=self.config['data_shape'][3], acti_func='prelu', dense_features=self.dense_features)

    def forward(self,x):
        x = self.in_conv0(x)
        #print(f'x {x.shape}')
        f1 = self.block1_1(x)
        #print(f'b1_1 {f1.shape}')
        f1 = self.block1_2(f1)
        #print(f'b1_2 {f1.shape}')
        f1 = self.fuse1(f1)
        #print(f'f1 {f1.shape}')
        if(self.downsample_twice):
            f1 = self.downsample1(f1)
            #print(f'downsample_twice {f1.shape}')
        if(self.base_chns[0] != self.base_chns[1]):
            f1 = self.feature_expand1(f1)
            #print(f'expand1 {f1.shape}')
        f1 = self.block2_1(f1)
        #print(f'b2_1 {f1.shape}')
        f1 = self.block2_2(f1)
        #print(f'b2_2 {f1.shape}')
        f1 = self.fuse2(f1)
        #print(f'f2 {f1.shape}')

        f2 = self.downsample2(f1)
        #print(f'downsample2_twice {f2.shape}')
        if(self.base_chns[1] != self.base_chns[2]):
            f2 = self.feature_expand2(f2)
            #print(f'expand2 {f2.shape}')
        f2 = self.block3_1(f2)
        #print(f'b3_1 {f2.shape}')
        f2 = self.block3_2(f2)
        #print(f'b3_2 {f2.shape}')
        f2 = self.block3_3(f2)
        #print(f'b3_3 {f2.shape}')
        f2 = self.fuse3(f2)
        #print(f'f3 {f2.shape}')
        
        f3 = f2
        if(self.base_chns[2] != self.base_chns[2]):
            f3 = self.feature_expand3(f3)
            #print(f'expand3 {f3.shape}') 
        f3 = self.block4_1(f3)
        #print(f'b4_1 {f3.shape}') 
        f3 = self.block4_2(f3)
        #print(f'b4_2 {f3.shape}')
        f3 = self.block4_3(f3)
        #print(f'b4_3 {f3.shape}')
        f3 = self.fuse4(f3)
        #print(f'f4 {f3.shape}')
        
        p1 = self.centra_slice1(f1)
        #print(f'p1_cs {p1.shape}')
        if(self.downsample_twice):
            p1 = self.pred_up1(p1)
            #print(f'p1_up1 {p1.shape}')
        else:
            p1 = self.pred1(p1)
         
        p2 = self.centra_slice2(f2)
        #print(f'p2_cs {p2.shape}')
        p2 = self.pred_up2_1(p2)
        #print(f'p2_up1 {p2.shape}')
        if(self.downsample_twice):
            p2 = self.pred_up2_2(p2)
            #print(f'p2_up2 {p2.shape}')
        
        p3 = self.pred_up3_1(f3)
        #print(f'p3_up1 {p3.shape}')
        if(self.downsample_twice):
            p3 = self.pred_up3_2(p3)
            #print(f'p3_up2 {p3.shape}')
        cat = torch.cat((p1, p2, p3), dim = 1)
        #print(f'cat {cat.shape}')
        pred = self.final_pred(cat)
        #print(f'pred {pred.shape}')

        if self.vae_enable: 
            out_vae, out_distr = self.vae(f3)
            out_final = torch.cat((pred, out_vae), 1)
            return out_final, out_distr

        return pred

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

class TensorSliceLayer(nn.Module):
    """
    extract the central part of a tensor
    """

    def __init__(self, margin = 1, regularizer=None, name='tensor_extract'):
        super(TensorSliceLayer, self).__init__()
        self.layer_name = name
        self.margin = margin
        
    def forward(self, x):
        x_shape = list(x.shape)
        begin = [0]*len(x_shape)
        begin[2] = self.margin
        dimention = 2
        size = x_shape
        length = size[2] - 2* self.margin
        output = torch.narrow(x,dimention,begin[2],length)
        return output

class ConvolutionalLayer(nn.Module):
    """
    This class defines a composite layer with optional components::

        convolution -> feature_normalization (default batch norm) -> activation -> dropout
    the feature normalization layer, and the activation layer (for 'prelu')
    Todo: add customized same padding if performs bad
    """
    def __init__(self, n_input_chns,
                n_output_chns,
                kernel_size=3,
                stride=1,
                dilation=1,
                with_bias=False,
                acti_func='prelu',
                momentum=0.1,
                eps=1e-5,
                padding=0,
                name="conv"):
        super(ConvolutionalLayer, self).__init__()
        
        # for ConvLayer
        self.name = name
        self.bn = nn.BatchNorm3d(n_input_chns, eps=eps, momentum=momentum)
        if acti_func == "relu":
            self.actv= nn.ReLU(inplace=True)
        if acti_func == "prelu":
            self.actv = nn.PReLU()
        self.conv = nn.Conv3d(in_channels=n_input_chns,
                            out_channels=n_output_chns,
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation,
                            padding=padding, 
                            bias=with_bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)

        return out


class DeconvolutionalLayer(nn.Module):
    """
    This class defines a composite layer with optional components::

        convolution -> feature_normalization (default batch norm) -> activation -> dropout
    the feature normalization layer, and the activation layer (for 'prelu')
    """

    def __init__(self, n_input_chns,
                n_output_chns,
                kernel_size=3,
                stride=1,
                with_bias=False,
                acti_func='prelu',
                momentum=0.1,
                eps=1e-5,
                padding=0,
                normalizaiton=None,
                num_groups=8,
                name="conv"):
        super(DeconvolutionalLayer, self).__init__()
        
        # for ConvLayer
        self.name = name
        if normalizaiton:
            self.bn =  nn.GroupNorm(num_groups=num_groups, num_channels=n_output_chns)
        else:
            self.bn = nn.BatchNorm3d(n_output_chns, eps=eps, momentum=momentum)
        if acti_func == "relu":
            self.actv= nn.ReLU(inplace=True)
        if acti_func == "prelu":
            self.actv = nn.PReLU()
        self.deconv = nn.ConvTranspose3d(in_channels=n_input_chns,
                            out_channels=n_output_chns,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=with_bias) 

    def forward(self, x):
        out = self.deconv(x)
        #out = F.pad(out, (0, 1, 0, 1, 0, 0))
        out = self.bn(out)
        out = self.actv(out)

        return out

class ResBlock(nn.Module):
    """
    This class define a high-resolution block with residual connections
    kernels - specify kernel sizes of each convolutional layer
            - e.g.: kernels=(5, 5, 5) indicate three conv layers of kernel_size 5
    with_res - whether to add residual connections to bypass the conv layers
    """
    def __init__(self,n_input_chns,
                 n_output_chns,
                 kernels=[(1, 3, 3), (1, 3, 3)],
                 strides=[(1, 1, 1), (1, 1, 1)],
                 dilation_rates=[(1, 1, 1), (1, 1, 1)],
                 paddings=[(0,1,1),(0,1,1)],
                 with_res=True,
                 use_bias=False,
                 acti_func="prelu",
                 name='ResBlock'):

        """Initializes ResBlock module."""
        super(ResBlock, self).__init__()
        self.with_res = with_res
        self.name = name
        self.bn1 = nn.BatchNorm3d(n_input_chns)
        self.bn2 = nn.BatchNorm3d(n_output_chns)
        #self.bn1 = nn.GroupNorm(8,n_input_chns)
        #self.bn2 = nn.GroupNorm(8,n_output_chns)
        if acti_func == "relu":
            self.actv= nn.ReLU(inplace=True)
        if acti_func == "prelu":
            self.actv = nn.PReLU()
        self.conv1 = nn.Conv3d(in_channels=n_input_chns,
                                out_channels=n_output_chns,
                                kernel_size=kernels[0],
                                stride=strides[0],
                                dilation=dilation_rates[0],
                                padding=paddings[0], 
                                bias=use_bias)

        self.conv2 = nn.Conv3d(in_channels=n_output_chns, #using the conv1 output chns
                        out_channels=n_output_chns,
                        kernel_size=kernels[1],
                        stride=strides[1],
                        dilation=dilation_rates[1],
                        padding=paddings[1],
                        bias=use_bias)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.actv(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.actv(out)
        out = self.conv2(out)
        # make residual connections
        if self.with_res:
            #residual = ElementwiseLayer('SUM', out, residual).layer_op()
            out += residual

        return out

class LinearUpSampling(nn.Module):
    '''
    Trilinear interpolate to upsampling
    '''
    def __init__(self, inChans, outChans, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
    
    def forward(self, x, skipx=None):
        out = self.conv1(x)
        # out = self.up1(out)
        out = nn.functional.interpolate(out, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        if skipx is not None:
            out = torch.cat((out, skipx), 1)
            out = self.conv2(out)
        
        return out

class VDResampling(nn.Module):
    '''
    Variational Auto-Encoder Resampling block
    '''
    def __init__(self, inChans=32, outChans=32, dense_features=(10,12,8), stride=2, kernel_size=3, padding=1, acti_func="relu", normalizaiton="group_normalization"):
        super(VDResampling, self).__init__()
        
        midChans = int(inChans / 2)
        self.dense_features = dense_features
        if normalizaiton == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=8,num_channels=inChans)
        if acti_func == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif acti_func == "prelu":
            self.actv1 = nn.PReLU()
            self.actv2 = nn.PReLU()
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dense1 = nn.Linear(in_features=16*dense_features[0]*dense_features[1]*dense_features[2], out_features=inChans)
        self.dense2 = nn.Linear(in_features=midChans, out_features=midChans*dense_features[0]*dense_features[1]*dense_features[2])
        self.up0 = LinearUpSampling(midChans,outChans)
        
    def forward(self, x):
        bs = x.shape[0]
        #print(f'vdrs x {x.shape}')
        out = self.gn1(x)
        #print(f'vdrs gn1 {out.shape}')
        out = self.actv1(out)
        #print(f'vdrs actv1 {out.shape}')
        out = self.conv1(out)
        #print(f'vdrs conv1 {out.shape}')
        out = out.view(-1, self.num_flat_features(out))
        #print(f'out view {out.shape}')
        out_vd = self.dense1(out)
        #print(f'dense1 {out_vd.shape}')
        distr = out_vd 
        out = VDraw(out_vd)
        #print(f'vdraw {out.shape}')
        out = self.dense2(out)
        #print(f'dense2 {out.shape}')
        out = self.actv2(out)
        #print(f'actv2 {out.shape}')
        out = out.view((bs, 16, self.dense_features[0],self.dense_features[1],self.dense_features[2]))
        #print(f'out view {out.shape}')
        out = self.up0(out)
        #print(f'up0  {out.shape}')
        
        return out, distr
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features

def VDraw(x):
    # Generate a Gaussian distribution with the given mean(128-d) and std(128-d)
    return torch.distributions.Normal(x[:,:16], x[:,16:]).sample()

class VDecoderBlock(nn.Module):
    '''
    Variational Decoder block
    '''
    def __init__(self, inChans, outChans, acti_func="prelu", normalizaiton="group_normalization", mode="trilinear"):
        super(VDecoderBlock, self).__init__()

        self.up0 = LinearUpSampling(inChans, outChans, mode=mode)
        self.block = DeconvolutionalLayer(outChans, outChans, acti_func=acti_func, normalizaiton=normalizaiton)
    
    def forward(self, x):
        out = self.up0(x)
        out = self.block(out)

        return out

class VAE(nn.Module):
    '''
    Variational Auto-Encoder : to group the features extracted by Encoder
    '''
    def __init__(self, inChans=32, outChans=1, dense_features=(10,12,8), acti_func="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VAE, self).__init__()

        self.vd_resample = VDResampling(inChans=inChans, outChans=inChans, acti_func=acti_func, dense_features=dense_features)
        self.vd_block2 = VDecoderBlock(inChans//4, inChans//4)
        self.vd_block1 = VDecoderBlock(inChans//4, inChans//4)
        self.vd_block0 = VDecoderBlock(inChans//8, inChans//8)
        self.vd_end = nn.Conv3d(inChans//8, outChans, kernel_size=1)
        
    def forward(self, x):
        out, distr = self.vd_resample(x)
        out = self.vd_block2(out)
        out = self.vd_block1(out)
        out = self.vd_block0(out)
        out = self.vd_end(out)

        return out, distr

class ElementwiseLayer(nn.Module):
    """
    This class takes care of the elementwise sum in a residual connection
    It matches the channel dims from two branch flows,
    by either padding or projection if necessary.
    """

    def __init__(self,
                 func,
                 input_block,
                 residual_block):
        super(ElementwiseLayer, self).__init__()
        self.func = func
        self.input_block = input_block
        self.residual_block = residual_block

    def layer_op(self):
        input_block_chns = self.input_block.shape[1]
        residual_block_chns = self.residual_block.shape[1]

        if self.func == 'SUM':
            if input_block_chns > residual_block_chns:  # pad the channel dim
                pad_1 = np.int((input_block_chns - residual_block_chns) // 2)
                zero_pads1 = torch.zeros(self.input_block.size(0), pad_1, self.input_block.size(2),
                        self.input_block.size(3), self.input_block.size(4))
                pad_2 = np.int(input_block_chns - residual_block_chns - pad_1)
                zero_pads2 = torch.zeros(self.input_block.size(0), pad_2, self.input_block.size(2),
                        self.input_block.size(3), self.input_block.size(4))
                if isinstance(self.input_block.data, torch.cuda.FloatTensor):
                    zero_pads1 = zero_pads1.cuda()
                    zero_pads2 = zero_pads2.cuda()

                self.residual_block = torch.cat([zero_pads1, self.residual_block.data, zero_pads2], dim=1)

            elif input_block_chns < residual_block_chns:  # make a projection
                projector = nn.Conv3d(residual_block_chns,
                                      input_block_chns,
                                      kernel_size=1,
                                      stride=1,
                                      )
                self.residual_block = projector(self.residual_block)

            # element-wise sum of both paths
            output_tensor = self.input_block + self.residual_block

        elif self.func == 'CONCAT':
            output_tensor =torch.cat([self.input_block, self.residual_block], dim=1)

        return output_tensor 

class DownSampling(nn.Module):
    # 3x3x3 convolution and 1 padding as default
    def __init__(self, inChans, outChans, stride=2, kernel_size=3, padding=1, dropout_rate=None):
        super(DownSampling, self).__init__()
        
        self.dropout_flag = False
        self.conv1 = nn.Conv3d(in_channels=inChans, 
                     out_channels=outChans, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding,
                     bias=False)
        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(dropout_rate,inplace=True)
            
    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_flag:
            out = self.dropout(out)
        return out

class OutputTransition(nn.Module):
    '''
    Decoder output layer 
    output the prediction of segmentation result
    '''
    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=(1, 3, 3), padding=(0,1,1))
        self.actv1 = torch.sigmoid
        
    def forward(self, x):
        return self.actv1(self.conv1(x))

