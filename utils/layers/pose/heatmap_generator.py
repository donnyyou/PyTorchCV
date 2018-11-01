#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch

from utils.tools.logger import Logger as Log


class HeatmapGenerator(object):

    def __init__(self, configer):
        self.configer = configer

    def __call__(self, gt_kpts, input_size, maskmap=None):
        width, height = input_size
        stride = self.configer.get('network', 'stride')
        num_keypoints = self.configer.get('data', 'num_kpts')
        sigma = self.configer.get('heatmap', 'sigma')
        method = self.configer.get('heatmap', 'method')
        batch_size = len(gt_kpts)

        heatmap = np.zeros((num_keypoints + 1, height // stride, width // stride), dtype=np.float32)
        start = stride / 2.0 - 0.5

        for i in range(len(gt_kpts)):
            for j in range(num_keypoints):
                if gt_kpts[i][j][2] < 0:
                    continue

                x = gt_kpts[i][j][0]
                y = gt_kpts[i][j][1]
                for h in range(height // stride):
                    for w in range(width // stride):
                        xx = start + w * stride
                        yy = start + h * stride
                        dis = 0
                        if method == 'gaussian':
                            dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                        elif method == 'laplace':
                            dis = math.sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma
                        else:
                            Log.error('Method: {} is not valid.'.format(method))
                            exit(1)

                        if dis > 4.6052:
                            continue

                            # Use max operator?
                        heatmap[j][h][w] = max(math.exp(-dis), heatmap[j][h][w])
                        if heatmap[j][h][w] > 1:
                            heatmap[j][h][w] = 1

            heatmap[num_keypoints, :, :] = 1.0 - np.max(heatmap[:-1, :, :], axis=0)

        heatmap = torch.from_numpy(heatmap)
        if maskmap is not None:
            heatmap = heatmap * maskmap.unsqueeze(0)

        return heatmap
