"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List,Optional,Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from CMRxRecon.models.unet import Unet

class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """
    def __init__(
            self,
            chans: int,
            num_pools: int,
            in_chans: int = 2,
            out_chans: int = 2,
            drop_prob: float = 0.0    
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layers.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability
        """
        super().__init__()

        self.unet = Unet(
            in_chans = in_chans,
            out_chans = out_chans,
            chans = chans,
            num_pool_layers = num_pools,
            drop_prob = drop_prob
        )
    
    def complex_to_chan_dim(self,x:torch.Tensor)->torch.Tensor:
        b,c,h,w,two = x.shape
        assert two == 2
        return x.permute(0,4,1,2,3).reshape(b,c*2,h,w)
    
    def chan_complex_to_last_dim(self,x:torch.Tensor)->torch.Tensor:
        b,c2,h,w = x.shape
        assert c2%2 == 0
        c = c2//2
        return x.view(b,2,c,h,w).permute(0,2,3,4,1).contiguous()
    
    def norm(self, x:torch.Tensor)->Tuple(torch.Tensor,torch.Tensor,torch.Tensor):
        #group norm
        b,c,h,w = x.shape
        x = x.view(b,2,c//2 * h * w)
        
        mean = x.mean(dim=2).view(b,2,1,1)
        std = x.std(dim=2).view(b,2,1,1)

        x = x.view(b,c,h,w)
        return (x-mean)/std, mean, std
    
    def unnorm(
            self, x:torch.Tensor, mean: torch.Tensor, std:torch.Tensor
    )->torch.Tensor:
        return x * std + mean
    
    def pad(
            self, x: torch.Tensor
    )->Tuple[torch.Tensor, Tuple[List[int],List[int],int,int]]:
        _,_,h,w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)
    
    def unpad(
            self,
            x: torch.Tensor,
            h_pad: List[int],
            w_pad: List[int],
            h_mult: int,
            w_mult: int
    ) -> torch.Tensor:
        return x[...,h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        
        #get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_size = self.pad(x)
        
        x = self.unet(x)

        #get shapes back and unnormalize
        x = self.unpad(x, *pad_size)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x
    
class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
            self,
            chans: int,
            num_pools: int,
            in_chans: int = 2,
            out_chans: int = 2,
            drop_prob: float = 0.0,
            mask_center: bool = True
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans = in_chans,
            out_chans = out_chans,
            drop_prob = drop_prob
        )
    


