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
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, img):
        img = Image.fromarray(img)
        assert isinstance(img, Image.Image)
        w, h = img.size

        pad = 4 * [None]
        pad[0] = 0  # left
        pad[1] = 0  # up
        pad[2] = 0 if (w % self.stride == 0) else self.stride - (w % self.stride)  # right
        pad[3] = 0 if (h % self.stride == 0) else self.stride - (h % self.stride)  # down

        img_padded = ImageOps.expand(img, tuple(pad), fill=0)  # confused.
        return img_padded, pad