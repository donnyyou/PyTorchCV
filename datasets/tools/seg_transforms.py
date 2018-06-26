#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import collections
from PIL import Image


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, Image.Image)

        if isinstance(self.size, int):
            w, h = img.size
            if (w >= h and w == self.size) or (h >= w and h == self.size):
                return img

            if w > h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), Image.BILINEAR)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), Image.BILINEAR)

        elif isinstance(self.size, collections.Iterable) and len(self.size) == 2:
            img = img.resize(self.size, Image.BILINEAR)
            return img

        else:
            raise TypeError('Got inappropriate size arg: {}'.format(self.size))


class ToLabel(object):
    def __call__(self, inputs):
        return torch.from_numpy(np.array(inputs)).long()


class ReLabel(object):
    """
      255 indicate the background, relabel 255 to some value.
    """
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        assert isinstance(inputs, torch.LongTensor), 'tensor needs to be LongTensor'

        inputs[inputs == self.olabel] = self.nlabel
        return inputs


class ToSP(object):
    def __init__(self, size):
        self.scale2 = Scale(size/2, Image.NEAREST)
        self.scale4 = Scale(size/4, Image.NEAREST)
        self.scale8 = Scale(size/8, Image.NEAREST)
        self.scale16 = Scale(size/16, Image.NEAREST)
        self.scale32 = Scale(size/32, Image.NEAREST)

    def __call__(self, input):
        input2 = self.scale2(input)
        input4 = self.scale4(input)
        input8 = self.scale8(input)
        input16 = self.scale16(input)
        input32 = self.scale32(input)
        inputs = [input, input2, input4, input8, input16, input32]
        # inputs = input

        return inputs


class Colorize(object):
    def __init__(self, color_list=None):
        self.cmap = color_list(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        # print size
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
