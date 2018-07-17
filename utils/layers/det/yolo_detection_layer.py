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


class YOLODetectionLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer):
        self.configer = configer
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')

    def __call__(self, layer_out, in_anchors, is_training=False):
        num_classes = self.configer.get('data', 'num_classes')
        if is_training:
            inp_dim = self.configer.get('data', 'train_input_size')
        else:
            inp_dim = self.configer.get('data', 'val_input_size')

        batch_size, _, grid_size_h, grid_size_w = layer_out.size()
        stride = inp_dim[0] / grid_size_w
        bbox_attrs = 4 + 1 + num_classes
        num_anchors = len(in_anchors)

        anchors = [(a[0] / stride, a[1] / stride) for a in in_anchors]

        layer_out = layer_out.view(batch_size, num_anchors * bbox_attrs, grid_size_h * grid_size_w)
        layer_out = layer_out.contiguous().view(batch_size, num_anchors, bbox_attrs, grid_size_h * grid_size_w)
        layer_out = layer_out.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, bbox_attrs)

        if not is_training:
            if self.configer.get('phase') == 'test':
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

            x_offset = x_offset.to(self.device)
            y_offset = y_offset.to(self.device)

            x_y_offset = torch.cat((x_offset, y_offset), 1).contiguous().view(1, -1, 2)
            x_y_offset = x_y_offset.repeat(num_anchors, 1, 1).view(-1, 2).unsqueeze(0)

            layer_out[:, :, :2] += x_y_offset

            # log space transform height and the width
            anchors = torch.FloatTensor(anchors)

            anchors = anchors.to(self.device)

            anchors = anchors.contiguous().view(3, 1, 2).repeat(1, grid_size_h * grid_size_w, 1).contiguous().view(-1, 2).unsqueeze(0)
            layer_out[:, :, 2:4] = torch.exp(layer_out[:, :, 2:4]) * anchors

            # Softmax the class scores
            if self.configer.get('phase') == 'test':
                layer_out[:, :, 5: 5 + num_classes] = torch.sigmoid((layer_out[:, :, 5: 5 + num_classes]))

            layer_out[:, :, 0] /= grid_size_w
            layer_out[:, :, 1] /= grid_size_h
            layer_out[:, :, 2] /= grid_size_w
            layer_out[:, :, 3] /= grid_size_h

        return layer_out