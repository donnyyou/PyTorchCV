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
        bboxes = torch.from_numpy(bboxes).float()

        return img, bboxes, labels
