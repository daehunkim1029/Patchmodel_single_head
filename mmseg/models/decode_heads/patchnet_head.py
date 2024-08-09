# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmseg.models.decode_heads.patch_decode_head import patch_BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

@MODELS.register_module()
class PatchnetHead(patch_BaseDecodeHead):
    """The all mlp Head of segformer.
    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.
    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """
    def __init__(self,
                 seg_head=None,
                 corruption_head=None,
                 interpolate_mode='bilinear',
                 conv_kernel_size=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = self.channels 
        self.interpolate_mode = interpolate_mode
        self.conv_kernel_size = conv_kernel_size
        self.conv_seg = None

        if seg_head:
            if self.conv_kernel_size == 1:
                self.seg_head = nn.Sequential(
                    nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size),
                    nn.SyncBatchNorm(self.input_dim//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
                )
            elif self.conv_kernel_size == 3:
                self.seg_head = nn.Sequential(
                    nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size, padding=1),
                    nn.SyncBatchNorm(self.input_dim//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
                )
        else:
            self.seg_head = None
                
        if corruption_head:
            if self.conv_kernel_size == 1:
                self.corruption_head = nn.Sequential(
                    nn.Conv2d(self.input_dim, self.input_dim // 2, kernel_size=self.conv_kernel_size, padding=1),
                    nn.SyncBatchNorm(self.input_dim//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.input_dim // 2, 1, kernel_size=1)  # Output 1 value per pixel
                )
            elif self.conv_kernel_size == 3:
                self.corruption_head = nn.Sequential(
                    nn.Conv2d(self.input_dim, self.input_dim // 2, kernel_size=self.conv_kernel_size, padding=1),
                    nn.SyncBatchNorm(self.input_dim//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.input_dim // 2, 1, kernel_size=1)  # Output 1 value per pixel
                )
        else:
            self.corruption_head = None

    def _forward_feature(self, inputs):
        feats = self._transform_inputs(inputs)
        return feats
    
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        x = inputs[-1]
        x = F.interpolate(x, size=(16, 16), mode=self.interpolate_mode)
        #import pdb; pdb.set_trace()
        corruption_outs = self.corruption_head(x) if self.corruption_head is not None else None
        seg_out = self.seg_head(x) if self.seg_head is not None else None
        
        return seg_out, corruption_outs