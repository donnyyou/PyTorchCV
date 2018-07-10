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
        self.aspect_ratios = 0

        # scale_w = self.configer.get('data', 'input_size')[0]
        # scale_h = self.configer.get('data', 'input_size')[1]
        # steps_w = [s / scale_w for s in self.configer.get('details', 'stride_list')] # modify
        # steps_h = [s / scale_h for s in self.configer.get('details', 'stride_list')]
        steps_w = [1.0 / f_hw[1] for f_hw in self.configer.get('details', 'feature_maps_hw')] # modify
        steps_h = [1.0 / f_hw[0] for f_hw in self.configer.get('details', 'feature_maps_hw')] # modify

        num_layers = len(self.configer.get('details', 'feature_maps_hw'))

        boxes = []
        for i in range(num_layers):
            fm_w = self.configer.get('details', 'feature_maps_hw')[i][1]
            fm_h = self.configer.get('details', 'feature_maps_hw')[i][0]
            for h, w in itertools.product(range(fm_h), range(fm_w)):
                cx = (w + 0.5) * steps_w[i]
                cy = (h + 0.5) * steps_h[i]

                if self.configer.get('details', 'anchor_method') == 'ssd':
                    s = self.configer.get('details', 'default_ratio_list')[i]
                    boxes.append((cx, cy, s, s))

                    s = math.sqrt(self.configer.get('details', 'default_ratio_list')[i]
                                  * self.configer.get('details', 'default_ratio_list')[i+1])

                    boxes.append((cx, cy, s, s))

                    s = self.configer.get('details', 'default_ratio_list')[i]
                    for ar in self.configer.get('details', 'aspect_ratio_list')[i]:
                        boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                        boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

                elif self.configer.get('details', 'anchor_method') == 'retina':
                    s = self.configer.get('details', 'default_ratio_list')[i]
                    for sr in self.configer.get('details', 'scale_ratio_list'):
                        s = sr * s
                        for ar in self.configer.get('details', 'aspect_ratio_list'):
                            boxes.append((cx, cy, s * ar, s / ar))

                else:
                    Log.error('Anchor Method {} not valid.'.format(self.configer.get('details', 'anchor_method')))
                    exit(1)

        boxes = np.array(boxes, dtype=float)
        boxes = torch.from_numpy(boxes).float()  # back to torch land
        if self.clip:
            boxes.clamp_(min=0., max=1.)

        return boxes


class FRPriorBoxLayer(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer, clip=True):
        self.configer = configer
        self.clip = clip

    def __call__(self):
        pass
