#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F

from utils.tools.logger import Logger as Log


class YOLODetectionLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer):
        self.configer = configer

    def __call__(self, layer_out, anchors, is_training=False):
        num_classes = self.configer.get('data', 'num_classes')
        if is_training:
            inp_dim = self.configer.get('data', 'train_input_size')
        else:
            inp_dim = self.configer.get('data', 'train_input_size')

        batch_size, _, grid_size_h, grid_size_w = layer_out.size()
        stride = inp_dim[0] / grid_size_w
        bbox_attrs = 4 + 1 + num_classes
        num_anchors = len(anchors)

        anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

        layer_out = layer_out.view(batch_size, num_anchors * bbox_attrs, grid_size_h * grid_size_w)
        layer_out = layer_out.transpose(1, 2).contiguous()
        layer_out = layer_out.view(batch_size, grid_size_h * grid_size_w * num_anchors, bbox_attrs)

        # Sigmoid the  centre_X, centre_Y. and object confidencce
        layer_out[:, :, 0] = torch.sigmoid(layer_out[:, :, 0])
        layer_out[:, :, 1] = torch.sigmoid(layer_out[:, :, 1])
        layer_out[:, :, 4] = torch.sigmoid(layer_out[:, :, 4])

        # Add the center offsets
        grid_len_h = np.arange(grid_size_h)
        grid_len_w = np.arange(grid_size_w)
        a, b = np.meshgrid(grid_len_w, grid_len_h)

        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)

        device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        x_offset = x_offset.to(device)
        y_offset = y_offset.to(device)

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

        layer_out[:, :, :2] += x_y_offset

        # log space transform height and the width
        anchors = torch.FloatTensor(anchors)

        anchors = anchors.to(device)

        anchors = anchors.repeat(grid_size_h * grid_size_w, 1).unsqueeze(0)
        layer_out[:, :, 2:4] = torch.exp(layer_out[:, :, 2:4]) * anchors

        # Softmax the class scores
        layer_out[:, :, 5: 5 + num_classes] = torch.sigmoid((layer_out[:, :, 5: 5 + num_classes]))

        layer_out[:, :, :4] *= stride

        return layer_out