#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch

from utils.tools.logger import Logger as Log


class FRPriorBoxLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer, clip=False):
        self.configer = configer
        self.clip = clip

    def __call__(self):
        img_w, img_h = self.configer.get('data', 'input_size')
        feature_map_w = [int(round(img_w / s)) for s in self.configer.get('rpn', 'stride_list')]
        feature_map_h = [int(round(img_h / s)) for s in self.configer.get('rpn', 'stride_list')]
        num_layers = len(self.configer.get('rpn', 'stride_list'))
        anchor_boxes_list = list()
        for i in range(num_layers):
            fm_w = feature_map_w[i]
            fm_h = feature_map_h[i]
            boxes = []
            anchor_sizes = self.configer.get('rpn', 'anchor_sizes_list')[i]
            for j in range(len(anchor_sizes)):
                s_w = anchor_sizes[j][0] / img_w
                s_h = anchor_sizes[j][1] / img_h
                boxes.append((0.5 / fm_w, 0.5 / fm_h, s_w, s_h))

                for ar in self.configer.get('rpn', 'aspect_ratio_list')[i]:
                    boxes.append((0.5 / fm_w, 0.5 / fm_h, s_w * math.sqrt(ar), s_h / math.sqrt(ar)))
                    boxes.append((0.5 / fm_w, 0.5 / fm_h, s_w / math.sqrt(ar), s_h * math.sqrt(ar)))

            anchor_bases = torch.from_numpy(np.array(boxes))
            assert anchor_bases.size(0) == self.configer.get('rpn', 'num_anchor_list')[i]
            anchors = anchor_bases.contiguous().view(1, -1, 4).repeat(fm_h * fm_w, 1, 1).contiguous().view(-1, 4)
            grid_len_h = np.arange(fm_h)
            grid_len_w = np.arange(fm_w)
            a, b = np.meshgrid(grid_len_w, grid_len_h)

            x_offset = torch.FloatTensor(a).view(-1, 1).div(fm_w)
            y_offset = torch.FloatTensor(b).view(-1, 1).div(fm_h)

            x_y_offset = torch.cat((x_offset, y_offset), 1).contiguous().view(-1, 1, 2)
            x_y_offset = x_y_offset.repeat(1, self.configer.get('rpn', 'num_anchor_list')[i], 1).contiguous().view(-1, 4)
            anchors[:, :2] = anchors[:, :2] + x_y_offset
            anchor_boxes_list.append(anchors)

        anchor_boxes = torch.cat(anchor_boxes_list, 0)
        if self.clip:
            anchor_boxes.clamp_(min=0., max=1.)

        return anchor_boxes
