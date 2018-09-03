#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from utils.helpers.image_helper import ImageHelper


class ResizeBoxes(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes, labels):
        width, height = ImageHelper.get_size(img)
        if bboxes is not None:
            bboxes[:, 0::2] /= width
            bboxes[:, 1::2] /= height

        labels = torch.from_numpy(labels).long()
        bboxes = torch.from_numpy(bboxes).float().clamp_(min=0.0, max=1.0)

        return img, bboxes, labels


class BoundResize(object):
    def __init__(self, resize_bound=(600, 1000)):
        self.resize_bound = resize_bound

    def __call__(self, img):
        img_size = ImageHelper.get_size(img)
        scale1 = self.resize_bound[0] / min(img_size)
        scale2 = self.resize_bound[1] / max(img_size)
        scale = min(scale1, scale2)
        target_size = [int(round(i*scale)) for i in img_size]
        img = ImageHelper.resize(img, target_size=target_size)
        return img, scale
