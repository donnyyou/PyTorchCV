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
        if bboxes is not None and len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i][0] /= img.size[0]
                bboxes[i][1] /= img.size[1]
                bboxes[i][2] /= img.size[0]
                bboxes[i][3] /= img.size[1]

            labels = torch.from_numpy(np.array(labels)).long()
            bboxes = torch.from_numpy(np.array(bboxes)).float()

        return img, bboxes, labels
