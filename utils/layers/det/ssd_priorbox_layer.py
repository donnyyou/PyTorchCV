#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

import numpy as np
import torch

from utils.tools.logger import Logger as Log


class SSDPriorBoxLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer, clip=True):
        self.configer = configer
        self.clip = clip

    def __call__(self):
        scale_w = self.configer.get('data', 'input_size')[0]
        scale_h = self.configer.get('data', 'input_size')[1]

        feature_map_w = [int(round(scale_w / s)) for s in self.configer.get('gt', 'stride_list')]
        feature_map_h = [int(round(scale_h / s)) for s in self.configer.get('gt', 'stride_list')]

        num_layers = len(self.configer.get('gt', 'stride_list'))
        prior_box_list = list()
        for i in range(num_layers):
            fm_w = feature_map_w[i]
            fm_h = feature_map_h[i]
            boxes = []
            if self.configer.get('gt', 'anchor_method') == 'ssd':
                s = self.configer.get('gt', 'default_ratio_list')[i]
                boxes.append((0.5 / fm_w, 0.5 / fm_h, s, s))

                s = math.sqrt(self.configer.get('gt', 'default_ratio_list')[i]
                              * self.configer.get('gt', 'default_ratio_list')[i+1])

                boxes.append((0.5 / fm_w, 0.5 / fm_h, s, s))

                s = self.configer.get('gt', 'default_ratio_list')[i]
                for ar in self.configer.get('gt', 'aspect_ratio_list')[i]:
                    boxes.append((0.5 / fm_w, 0.5 / fm_h, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((0.5 / fm_w, 0.5 / fm_h, s / math.sqrt(ar), s * math.sqrt(ar)))

            elif self.configer.get('gt', 'anchor_method') == 'retina':
                s = self.configer.get('gt', 'default_ratio_list')[i]
                for sr in self.configer.get('gt', 'scale_ratio_list'):
                    s = sr * s
                    for ar in self.configer.get('gt', 'aspect_ratio_list'):
                        boxes.append((0.5 / fm_w, 0.5 / fm_h, s * ar, s / ar))

                else:
                    Log.error('Anchor Method {} not valid.'.format(self.configer.get('gt', 'anchor_method')))
                    exit(1)

            anchor_bases = torch.from_numpy(np.array(boxes))
            assert anchor_bases.size(0) == self.configer.get('gt', 'num_anchor_list')[i]
            anchors = anchor_bases.contiguous().view(1, -1, 4).repeat(fm_h * fm_w, 1, 1).contiguous().view(-1, 4)
            grid_len_h = np.arange(fm_h)
            grid_len_w = np.arange(fm_w)
            a, b = np.meshgrid(grid_len_w, grid_len_h)

            x_offset = torch.FloatTensor(a).view(-1, 1).div(fm_w)
            y_offset = torch.FloatTensor(b).view(-1, 1).div(fm_h)

            x_y_offset = torch.cat((x_offset, y_offset), 1).contiguous().view(-1, 1, 2)
            x_y_offset = x_y_offset.repeat(1, self.configer.get('gt', 'num_anchor_list')[i], 1).contiguous().view(-1, 2)
            anchors = anchors[:, :2] * x_y_offset
            prior_box_list.append(anchors)

        anchor_boxes = torch.cat(prior_box_list, 0)
        if self.clip:
            anchor_boxes.clamp_(min=0., max=1.)

        return anchor_boxes

