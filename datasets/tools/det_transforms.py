#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from PIL import Image


class ResizeBoxes(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes, labels):
        assert isinstance(img, Image.Image)
        if bboxes is not None:
            bboxes[:, 0::2] /= img.size[0]
            bboxes[:, 1::2] /= img.size[1]

        labels = torch.from_numpy(labels).long()
        bboxes = torch.from_numpy(bboxes).float()

        return img, bboxes, labels
