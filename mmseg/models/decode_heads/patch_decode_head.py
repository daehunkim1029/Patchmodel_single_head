# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..losses import accuracy
from ..utils import resize

class patch_BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead."""

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 corruption_threshold=0.3,
                 corruption_step=2,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super().__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.corruption_threshold = corruption_threshold
        self.corruption_step = corruption_step

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert `seg_logits` into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.out_channels = out_channels
        self.threshold = threshold

        if isinstance(loss_cls, dict):
            self.loss_cls = MODELS.build(loss_cls)
        elif isinstance(loss_cls, (list, tuple)):
            self.loss_cls = nn.ModuleList()
            for loss in loss_cls:
                self.loss_cls.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_cls must be a dict or sequence of dict,\
                but got {type(loss_cls)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms."""
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder."""
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training."""
        results = self.forward(inputs)        
        losses = self.total_loss(results, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction."""
        return self.forward(inputs)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def viz_seg(self, arr):
        arr = arr.cpu().numpy()
        cmap_binary = mcolors.ListedColormap(['white', 'blue'])
        cmap_custom = mcolors.ListedColormap(['white', 'blue', 'black'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.imshow(arr[0], cmap=cmap_binary, vmin=0, vmax=1)
        ax1.set_title('[0, :, :]')
        ax1.axis('off')

        ax2.imshow(arr[1], cmap=cmap_custom, vmin=0, vmax=2)
        ax2.set_title('[1, :, :]')
        ax2.axis('off')

        cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_binary), ax=ax1, ticks=[0, 1], orientation='horizontal')
        cbar1.set_ticklabels(['White (0)', 'Blue (1)'])

        cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_custom), ax=ax2, ticks=[0, 1, 2], orientation='horizontal')
        cbar2.set_ticklabels(['White (0)', 'Blue (1)', 'Black (2)'])

        plt.tight_layout()
        plt.savefig('visualization_after.png', dpi=300)

    def total_loss(self, results: Tensor,
                   batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss."""
        gt_labels = self._stack_batch_gt(batch_data_samples)
        loss = dict()

        if gt_labels.sum() == 0:
            seg_label = torch.zeros_like(gt_labels)
            corruption_label = torch.zeros_like(gt_labels)
        else:
            prepare_for_seg_label = gt_labels.clone()
            prepare_for_seg_label[prepare_for_seg_label != 0] += 1
            seg_label = torch.div(prepare_for_seg_label, 2, rounding_mode='trunc')
            prepare_for_corruption_label = seg_label.clone()
            prepare_for_corruption_label[prepare_for_corruption_label != 0] -= 1
            corruption_label = gt_labels - prepare_for_corruption_label*2
            corruption_label = corruption_label / self.corruption_step

        seg_out, corruption_outs = results

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_out, seg_label)
        else:
            seg_weight = None

        if corruption_outs is not None:
            loss['loss_corruption'] = F.smooth_l1_loss(corruption_outs.sigmoid(), corruption_label)

        if seg_out is not None:
            seg_GT_mask = (corruption_outs > self.corruption_threshold).long().view(-1, 1, 16, 16).contiguous()
            seg_pred_mask = (corruption_outs > self.corruption_threshold).long().repeat(1, self.num_classes, 1, 1).contiguous()

            seg_GT = (seg_label*seg_GT_mask).squeeze(1)
            seg_pred = (seg_out*seg_pred_mask).squeeze(1)
            seg_GT = torch.where(seg_GT_mask.squeeze(1) == 0, torch.tensor(255, dtype=seg_GT.dtype, device=seg_GT.device), seg_GT)
            seg_pred = torch.where(seg_pred_mask.squeeze(1) == 0, torch.tensor(255, dtype=seg_pred.dtype, device=seg_pred.device), seg_pred)

            if not isinstance(self.loss_cls, nn.ModuleList):
                losses_decode = [self.loss_cls]
            else:
                losses_decode = self.loss_cls
            for loss_cls in losses_decode:
                if loss_cls.loss_name not in loss:
                    loss[loss_cls.loss_name] = loss_cls(
                        seg_pred,
                        seg_GT,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_cls.loss_name] += loss_cls(
                        seg_pred,
                        seg_GT,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape."""
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits