#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Utilizer class for dataset loader.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch

from utils.tools.logger import Logger as Log


class PoseDataUtilizer(object):

    def __init__(self, configer):
        self.configer = configer

    def generate_heatmap(self, kpts=None, mask=None):

        height = self.configer.get('data', 'input_size')[1]
        width = self.configer.get('data', 'input_size')[0]
        stride = self.configer.get('network', 'stride')
        num_keypoints = self.configer.get('data', 'num_keypoints')
        sigma = self.configer.get('heatmap', 'sigma')
        method = self.configer.get('heatmap', 'method')

        heatmap = np.zeros((height // stride,
                            width // stride,
                            num_keypoints + 1), dtype=np.float32)
        start = stride / 2.0 - 0.5
        num_objects = len(kpts)
        for i in range(num_objects):
            for j in range(num_keypoints):
                if kpts[i][j][2] < 0:
                    continue

                x = kpts[i][j][0]
                y = kpts[i][j][1]
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
                        heatmap[h][w][j] = max(math.exp(-dis), heatmap[h][w][j])
                        if heatmap[h][w][j] > 1:
                            heatmap[h][w][j] = 1

        heatmap[:, :, num_keypoints] = 1.0 - np.max(heatmap[:, :, :-1], axis=2)
        if mask is not None:
            heatmap = heatmap * mask

        return heatmap

    def generate_tagmap(self, kpts=None,):
        height = self.configer.get('data', 'input_size')[1]
        width = self.configer.get('data', 'input_size')[0]
        num_keypoints = self.configer.get('data', 'num_keypoints')
        stride = self.configer.get('network', 'stride')

        tagmap = np.zeros((num_keypoints, height // stride, width // stride), dtype=np.float32)
        num_objects = 0
        for i in range(len(kpts)):
            flag = False
            for j in range(num_keypoints):
                if kpts[i][j][2] < 0:
                    continue

                if kpts[i][j][0] < 0 or kpts[i][j][1] < 0:
                    continue

                if kpts[i][j][0] >= width or kpts[i][j][1] >= height:
                    continue

                if flag == False:
                    num_objects += 1
                    flag = True

                tagx = int(kpts[i][j][0] // stride)
                tagy = int(kpts[i][j][1] // stride)
                tagmap[j][tagy][tagx] = num_objects

        tagmap = torch.FloatTensor(tagmap)
        num_objects = torch.IntTensor([num_objects])
        return tagmap, num_objects

    def generate_paf(self, kpts=None, mask=None):
        vec_pair = self.configer.get('details', 'limb_seq')
        height = self.configer.get('data', 'input_size')[1]
        width = self.configer.get('data', 'input_size')[0]
        stride = self.configer.get('network', 'stride')
        theta = self.configer.get('heatmap', 'theta')
        vecmap = np.zeros((height // stride, width // stride, len(vec_pair) * 2), dtype=np.float32)
        cnt = np.zeros((height // stride, width // stride, len(vec_pair)), dtype=np.int32)

        height, width, channel = cnt.shape
        num_objects = len(kpts)

        start = stride / 2.0 - 0.5

        for j in range(num_objects):
            for i in range(channel):
                a = vec_pair[i][0] - 1
                b = vec_pair[i][1] - 1
                if kpts[j][a][2] < 0 or kpts[j][b][2] < 0:
                    continue

                ax = (kpts[j][a][0] - start) * 1.0 / stride
                ay = (kpts[j][a][1] - start) * 1.0 / stride
                bx = (kpts[j][b][0] - start) * 1.0 / stride
                by = (kpts[j][b][1] - start) * 1.0 / stride

                bax = bx - ax
                bay = by - ay
                # 1e-9 to aviod two points have same position.
                norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
                bax /= norm_ba
                bay /= norm_ba

                min_w = max(int(round(min(ax, bx) - theta)), 0)
                max_w = min(int(round(max(ax, bx) + theta)), width-1)
                min_h = max(int(round(min(ay, by) - theta)), 0)
                max_h = min(int(round(max(ay, by) + theta)), height-1)

                for h in range(min_h, max_h+1):
                    for w in range(min_w, max_w+1):
                        px = w - ax
                        py = h - ay

                        dis = abs(bay * px - bax * py)
                        if dis <= theta:
                            vecmap[h][w][2*i] = (vecmap[h][w][2*i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                            vecmap[h][w][2*i+1] = (vecmap[h][w][2*i+1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                            cnt[h][w][i] += 1

        if mask is not None:
            vecmap = vecmap * mask

        return vecmap

    def generate_partmap(self, kpts=None, mask=None):
        vec_pair = self.configer.get('details', 'limb_seq')
        height = self.configer.get('data', 'input_size')[1]
        width = self.configer.get('data', 'input_size')[0]
        stride = self.configer.get('network', 'stride')
        sigma = self.configer.get('heatmap', 'sigma')
        method = self.configer.get('heatmap', 'method')
        theta = self.configer.get('heatmap', 'theta')
        partmap = np.zeros((height // stride, width // stride, len(vec_pair) * 6), dtype=np.float32)
        cnt = np.zeros((height // stride, width // stride, len(vec_pair)), dtype=np.int32)

        num_objects = len(kpts)

        start = stride / 2.0 - 0.5

        for j in range(num_objects):
            for i in range(len(vec_pair)):
                a = vec_pair[i][0] - 1
                b = vec_pair[i][1] - 1
                if kpts[j][a][2] < 0 or kpts[j][b][2] < 0:
                    continue

                ax = (kpts[j][a][0] - start) * 1.0 / stride
                ay = (kpts[j][a][1] - start) * 1.0 / stride
                bx = (kpts[j][b][0] - start) * 1.0 / stride
                by = (kpts[j][b][1] - start) * 1.0 / stride

                bax = bx - ax
                bay = by - ay
                # 1e-9 to aviod two points have same position.
                norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
                bax /= norm_ba
                bay /= norm_ba

                min_w = max(int(round(min(ax, bx) - theta)), 0)
                max_w = min(int(round(max(ax, bx) + theta)), width // stride - 1)
                min_h = max(int(round(min(ay, by) - theta)), 0)
                max_h = min(int(round(max(ay, by) + theta)), height // stride - 1)

                for h in range(min_h, max_h+1):
                    for w in range(min_w, max_w+1):
                        px = w - ax
                        py = h - ay

                        dis = abs(bay * px - bax * py)
                        if dis <= theta:
                            partmap[h][w][6*i] = (partmap[h][w][6*i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                            partmap[h][w][6*i+1] = (partmap[h][w][6*i+1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                            cnt[h][w][i] += 1

                for h in range(height // stride):
                    for w in range(width // stride):
                        xx = start + w * stride
                        yy = start + h * stride
                        x = start + ax * stride
                        y = start + ay * stride
                        dis = 0
                        if method == 'gaussian':
                            dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                        elif method == 'laplace':
                            dis = math.sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma
                        else:
                            Log.error('Method: {} is not valid.'.format(method))
                            exit(1)

                        if dis < 4.6052:
                            arrow_length = math.exp(-dis)
                            if arrow_length > math.sqrt(partmap[h][w][i*6+2]**2 + partmap[h][w][i*6+3]**2):
                                partmap[h][w][i*6+2] = bax * math.sqrt(arrow_length)
                                partmap[h][w][i*6+3] = bay * math.sqrt(arrow_length)

                        xx = start + w * stride
                        yy = start + h * stride
                        x = start + bx * stride
                        y = start + by * stride
                        dis = 0
                        if method == 'gaussian':
                            dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                        elif method == 'laplace':
                            dis = math.sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma
                        else:
                            Log.error('Method: {} is not valid.'.format(method))
                            exit(1)

                        if dis < 4.6052:
                            arrow_length = math.exp(-dis)
                            if arrow_length > math.sqrt(partmap[h][w][i*6+4]**2 + partmap[h][w][i*6+5]**2):
                                partmap[h][w][i*6+4] = -bax * math.sqrt(arrow_length)
                                partmap[h][w][i*6+5] = -bay * math.sqrt(arrow_length)

        if mask is not None:
            partmap = partmap * mask

        return partmap
