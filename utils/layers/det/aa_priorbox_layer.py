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


class AAPriorBoxLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer, clip=True):
        self.configer = configer
        self.clip = clip

    def __call__(self, feat_list, out_list, input_size):
        img_w, img_h = input_size
        feature_map_w = [feat.size(3) for feat in feat_list]
        feature_map_h = [feat.size(2) for feat in feat_list]
        stride_w_list = [img_w / feat_w for feat_w in feature_map_w]
        stride_h_list = [img_h / feat_h for feat_h in feature_map_h]

        anchor_boxes_list = list()
        for b in range(out_list[0].size(0)):
            b_anchors = []
            for i in range(len(feat_list)):
                stride_offset_w, stride_offset_h = 0.5 * stride_w_list[i], 0.5 * stride_h_list[i]
                s = self.configer.get('gt', 'cur_anchor_sizes')[i]
                anchor_bases = torch.ones_like(out_list[i][b]).mul_(s)
                anchor_bases = anchor_bases * out_list[i][b]
                grid_len_h = np.arange(0, img_h - stride_offset_h, stride_h_list[i])
                grid_len_w = np.arange(0, img_w - stride_offset_w, stride_w_list[i])
                a, b = np.meshgrid(grid_len_w, grid_len_h)

                x_offset = torch.FloatTensor(a).view(-1, 1)
                y_offset = torch.FloatTensor(b).view(-1, 1)

                x_y_offset = torch.cat((x_offset, y_offset), 1).contiguous().view(-1, 1, 2)
                x_y_offset = x_y_offset.repeat(1, self.configer.get('gt', 'num_anchor_list')[i], 1).contiguous().view(-1, 2)
                anchors = torch.cat((x_y_offset, anchor_bases), 1)
                b_anchors.append(anchors)

            anchors = torch.cat(b_anchors, 0)
            anchor_boxes_list.append(anchors)

        anchor_boxes = torch.stack(anchor_boxes_list, 0)
        if self.clip:
            anchor_boxes[:, 0::2].clamp_(min=0., max=img_w - 1)
            anchor_boxes[:, 1::2].clamp_(min=0., max=img_h - 1)

        return anchor_boxes
