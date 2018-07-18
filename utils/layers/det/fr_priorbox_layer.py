#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from utils.tools.logger import Logger as Log


class FRPriorBoxLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer, clip=True):
        self.configer = configer
        self.clip = clip

    def __call__(self):
        num_layers = len(self.configer.get('gt', 'feature_maps_hw'))
        prior_box_list = list()
        for layer_num in range(num_layers):
            fm_w = self.configer.get('gt', 'feature_maps_wh')[layer_num][0]
            fm_h = self.configer.get('gt', 'feature_maps_wh')[layer_num][1]
            stride = self.configer.get('gt', 'stride_list')[layer_num]
            anchor_size = self.configer.get('gt', 'anchor_size_list')[layer_num]
            aspect_ratio = self.configer.get('gt', 'aspect_ratio_list')[layer_num]
            anchor_bases = np.zeros((len(anchor_size) * len(aspect_ratio), 4), dtype=np.float32)
            for i in range(len(aspect_ratio)):
                for j in range(len(anchor_size)):
                    h = anchor_size[j] * np.sqrt(aspect_ratio[i])
                    w = anchor_size[j] * np.sqrt(1. / aspect_ratio[i])

                    index = i * len(anchor_size) + j
                    anchor_bases[index, 0] = stride // 2 - h / 2.
                    anchor_bases[index, 1] = stride // 2 - w / 2.
                    anchor_bases[index, 2] = stride // 2 + h / 2.
                    anchor_bases[index, 3] = stride // 2 + w / 2.

            shift_y = np.arange(0, fm_h * stride, stride)
            shift_x = np.arange(0, fm_w * stride, stride)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)
            A = anchor_bases.shape[0]
            K = shift.shape[0]
            anchor = anchor_bases.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
            anchor = anchor.reshape((K * A, 4)).astype(np.float32)
            prior_box_list.append(anchor)

        return torch.cat(prior_box_list, 0)
