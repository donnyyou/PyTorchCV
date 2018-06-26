#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai Li(xiangtai94@gmail.com)
# this scripts contain flow operation with torch library

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.autograd import Variable

class ImageWarp(object):
    def __init__(self):
        pass

    def __call__(self, image, flow, direction=1):
        """
            :param image: input image for warp
            :param flow: flow field
            :param direction:
            :return:
            """
        n, c, h, w = image.size()
        x = Variable(torch.arange(0, w).expand([n, h, w])).cuda()  # (n,h,w)
        y = Variable(torch.arange(0, h).expand([n, w, h])).permute(0, 2, 1).cuda()

        grid_x = (x + flow[:, 0, :, :] * direction) * 2 / (w - 1) - 1.0
        grid_y = (y + flow[:, 1, :, :] * direction) * 2 / (h - 1) - 1.0

        grid_xy = torch.cat([torch.unsqueeze(grid_x, 3), torch.unsqueeze(grid_y, 3)], 3)  # c,h,w,n

        warp_image = F.grid_sample(image, grid_xy)

        return warp_image