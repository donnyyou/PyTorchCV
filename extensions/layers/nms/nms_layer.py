#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# NMS layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import extensions.layers.nms.src.cython_nms as cython_nms


class NMS(object):
    def __init__(self):
        pass

    def __call__(self, dets, threshold):
        """Apply classic DPM-style greedy NMS."""
        if dets.shape[0] == 0:
            return []

        return cython_nms.nms(dets, threshold)


class SoftNMS(object):
    def __init__(self, sigma=0.5, method='linear'):
        self.sigma = sigma
        self.method = method

    def __call__(self, dets, overlap_threshold=0.3, score_threshold = 0.001):
        """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
        if dets.shape[0] == 0:
            return dets, []

        methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
        assert self.method in methods, 'Unknown soft_nms method: {}'.format(self.method)

        dets, keep = cython_nms.soft_nms(
            np.ascontiguousarray(dets, dtype=np.float32),
            np.float32(self.sigma),
            np.float32(overlap_threshold),
            np.float32(score_threshold),
            np.uint8(methods[self.method])
        )
        return dets, keep
