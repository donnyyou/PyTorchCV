#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Reweight(nn.Module):
    def __init__(self, spatial_size=4):
        super(Reweight, self).__init__()
        self.spatial_size = spatial_size

    def forward(self, x):
        pooled_x = F.adaptive_max_pool2d(x, output_size=self.spatial_size)
        b, c, h, w = pooled_x.size()
        mat1_x = pooled_x.contiguous().view(b, c, h*w)
        mat2_x = pooled_x.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)
        sim_mat = torch.matmul(mat1_x, mat2_x)
        weight = sim_mat.sum(dim=2).div(c).contigous().view(b, c, 1, 1)
        weight = 1 - weight
        out = weight * x
        return out

