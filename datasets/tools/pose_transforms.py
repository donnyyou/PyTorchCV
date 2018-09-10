#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image, ImageOps


class PadImage(object):
    """ Padding the Image to proper size.
        Args:
            stride: the stride of the network.
            pad_value: the value that pad to the image border.
            img: np.array object as input.

        Returns:
            img: np.array object.
    """
    def __init__(self, stride, mean_value=(104, 117, 123)):
        self.stride = stride
        self.mean_value = tuple(mean_value)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            h, w, c = img.shape

        pad = 4 * [None]
        pad[0] = 0  # left
        pad[1] = 0  # up
        pad[2] = 0 if (w % self.stride == 0) else self.stride - (w % self.stride)  # right
        pad[3] = 0 if (h % self.stride == 0) else self.stride - (h % self.stride)  # down

        if isinstance(img, Image.Image):
            img_padded = ImageOps.expand(img, tuple(pad), fill=self.mean_value)  # confused.
        else:
            img_padded = np.zeros((h + pad[3], w + pad[2], c), dtype=img.dtype)
            img_padded[:, :, :] = self.mean_value
            img_padded[:h, :w, :] = img

        return img_padded, pad