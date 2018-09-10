#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import torch

from utils.tools.logger import Logger as Log


class YOLODetectionLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer):
        self.configer = configer
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')

    def __call__(self, layer_out_list):
        num_classes = self.configer.get('data', 'num_classes')
        detect_list = list()
        prediction_list = list()
        for i in range(len(layer_out_list)):
            batch_size, _, grid_size_h, grid_size_w = layer_out_list[i].size()
            feat_stride = self.configer.get('network', 'stride_list')[i]
            in_anchors = self.configer.get('gt', 'anchors_list')[i]
            bbox_attrs = 4 + 1 + num_classes
            num_anchors = len(in_anchors)

            anchors = [(a[0] / feat_stride, a[1] / feat_stride) for a in in_anchors]

            layer_out = layer_out_list[i].view(batch_size, num_anchors * bbox_attrs, grid_size_h * grid_size_w)
            layer_out = layer_out.contiguous().view(batch_size, num_anchors, bbox_attrs, grid_size_h * grid_size_w)
            layer_out = layer_out.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, bbox_attrs)

            if self.configer.get('phase') != 'debug':
                # Sigmoid the  centre_X, centre_Y. and object confidencce
                layer_out[:, :, 0] = torch.sigmoid(layer_out[:, :, 0])
                layer_out[:, :, 1] = torch.sigmoid(layer_out[:, :, 1])
                layer_out[:, :, 4] = torch.sigmoid(layer_out[:, :, 4])

                # Softmax the class scores
                layer_out[:, :, 5: 5 + num_classes] = torch.sigmoid((layer_out[:, :, 5: 5 + num_classes]))

            prediction_list.append(layer_out)

            detect_out = layer_out.clone()
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

            detect_out[:, :, :2] += x_y_offset

            # log space transform height and the width
            anchors = torch.FloatTensor(anchors)

            anchors = anchors.to(self.device)

            anchors = anchors.contiguous().view(3, 1, 2)\
                .repeat(1, grid_size_h * grid_size_w, 1).contiguous().view(-1, 2).unsqueeze(0)
            detect_out[:, :, 2:4] = torch.exp(detect_out[:, :, 2:4]) * anchors

            detect_out[:, :, 0] /= grid_size_w
            detect_out[:, :, 1] /= grid_size_h
            detect_out[:, :, 2] /= grid_size_w
            detect_out[:, :, 3] /= grid_size_h

            detect_list.append(detect_out)

        return layer_out_list, torch.cat(prediction_list, 1), torch.cat(detect_list, 1)
