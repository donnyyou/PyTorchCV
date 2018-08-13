#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn


class ROIPoolingLayer(object):

    def __init__(self, configer):
        self.configer = configer
        self.pooled_width = int(self.configer.get('roi', 'pooled_width'))
        self.pooled_height = int(self.configer.get('roi', 'pooled_height'))
        self.spatial_scale = 1.0 / float(self.configer.get('roi', 'spatial_stride'))
        from extensions.layers.roi.roi_pool import _RoIPooling
        self.roi_pooling = _RoIPooling(self.pooled_height, self.pooled_width, self.spatial_scale)

    def __call__(self, features, rois):
        return self.roi_pooling(features, rois)


class PyROIPoolingLayer(nn.Module):
    def __init__(self, configer):
        super(PyROIPoolingLayer, self).__init__()
        self.configer = configer
        self.pooled_width = int(self.configer.get('roi', 'pooled_width'))
        self.pooled_height = int(self.configer.get('roi', 'pooled_height'))
        self.spatial_scale = 1.0 / float(self.configer.get('roi', 'spatial_stride'))

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]

        outputs = torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width).cuda()

        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].item())
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(
                roi[1:].data.cpu().numpy() * self.spatial_scale).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    is_empty = (hend <= hstart) or(wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        outputs[roi_ind, :, ph, pw] = torch.max(
                            torch.max(data[:, hstart:hend, wstart:wend], 1)[0], 1)[0].view(-1)

        return outputs