#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

ROI_POOL = True
ROI_ALIGN = True
try:
    from extensions.layers.roipool.module import RoIPool2D
except:
    ROI_POOL = False

try:
    from extensions.layers.roialign.module import RoIAlign2D
except:
    ROI_ALIGN = False

from utils.tools.logger import Logger as Log


class FRRoiProcessLayer(nn.Module):
    def __init__(self, configer):
        super(FRRoiProcessLayer, self).__init__()
        self.configer = configer

    def forward(self, features, rois, spatial_scale):
        out = None
        if self.configer.get('roi', 'method') == 'roipool':
            assert ROI_POOL is True
            out = RoIPool2D(pooled_height=int(self.configer.get('roi', 'pooled_height')),
                            pooled_width=int(self.configer.get('roi', 'pooled_width')),
                            spatial_scale=spatial_scale)(features, rois)

        elif self.configer.get('roi', 'method') == 'roialign':
            assert ROI_ALIGN is True
            out = RoIAlign2D(pooled_height=int(self.configer.get('roi', 'pooled_height')),
                             pooled_width=int(self.configer.get('roi', 'pooled_width')),
                             spatial_scale=spatial_scale,
                             sampling_ratio=2)

        else:
            Log.error('Invalid roi method.')
            exit(1)

        return out

