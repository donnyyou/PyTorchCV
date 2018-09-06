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


class SSDPriorBoxLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer, clip=True):
        self.configer = configer
        self.clip = clip

    def __call__(self, feat_list, input_size):
        img_w, img_h = input_size
        feature_map_w = [feat.size(3) for feat in feat_list]
        feature_map_h = [feat.size(2) for feat in feat_list]

        anchor_boxes_list = list()
        for i in range(len(feat_list)):
            fm_w = feature_map_w[i]
            fm_h = feature_map_h[i]
            boxes = []
            if self.configer.get('gt', 'anchor_method') == 'ssd':
                s_w = self.configer.get('gt', 'cur_anchor_sizes')[i] / img_w
                s_h = self.configer.get('gt', 'cur_anchor_sizes')[i] / img_h
                boxes.append((0.5 / fm_w, 0.5 / fm_h, s_w, s_h))

                base_s = math.sqrt(self.configer.get('gt', 'cur_anchor_sizes')[i]
                                   * self.configer.get('gt', 'cur_anchor_sizes')[i+1])

                s_w, s_h = base_s / img_w, base_s / img_h
                boxes.append((0.5 / fm_w, 0.5 / fm_h, s_w, s_h))

                s_w = self.configer.get('gt', 'cur_anchor_sizes')[i] / img_w
                s_h = self.configer.get('gt', 'cur_anchor_sizes')[i] / img_h
                for ar in self.configer.get('gt', 'aspect_ratio_list')[i]:
                    boxes.append((0.5 / fm_w, 0.5 / fm_h, s_w * math.sqrt(ar), s_h / math.sqrt(ar)))
                    boxes.append((0.5 / fm_w, 0.5 / fm_h, s_w / math.sqrt(ar), s_h * math.sqrt(ar)))

            elif self.configer.get('gt', 'anchor_method') == 'retina':
                s_w = self.configer.get('gt', 'cur_anchor_sizes')[i] / img_w
                s_h = self.configer.get('gt', 'cur_anchor_sizes')[i] / img_h
                for sr in self.configer.get('gt', 'scale_ratio_list'):
                    s_w = sr * s_w
                    s_h = sr * s_h
                    for ar in self.configer.get('gt', 'aspect_ratio_list'):
                        boxes.append((0.5 / fm_w, 0.5 / fm_h, s_w * ar, s_h / ar))

                else:
                    Log.error('Anchor Method {} not valid.'.format(self.configer.get('gt', 'anchor_method')))
                    exit(1)

            anchor_bases = torch.FloatTensor(np.array(boxes))
            assert anchor_bases.size(0) == self.configer.get('gt', 'num_anchor_list')[i]
            anchors = anchor_bases.contiguous().view(1, -1, 4).repeat(fm_h * fm_w, 1, 1).contiguous().view(-1, 4)
            grid_len_h = np.arange(fm_h)
            grid_len_w = np.arange(fm_w)
            a, b = np.meshgrid(grid_len_w, grid_len_h)

            x_offset = torch.FloatTensor(a).view(-1, 1).div(fm_w)
            y_offset = torch.FloatTensor(b).view(-1, 1).div(fm_h)

            x_y_offset = torch.cat((x_offset, y_offset), 1).contiguous().view(-1, 1, 2)
            x_y_offset = x_y_offset.repeat(1, self.configer.get('gt', 'num_anchor_list')[i], 1).contiguous().view(-1, 2)
            anchors[:, :2] = anchors[:, :2] + x_y_offset
            anchor_boxes_list.append(anchors)

        anchor_boxes = torch.cat(anchor_boxes_list, 0)
        if self.clip:
            anchor_boxes.clamp_(min=0., max=1.)

        return anchor_boxes

