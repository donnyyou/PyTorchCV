#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from PIL import Image
import collections


class Normalize(object):
    """Normalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        inputs = inputs.div(255)
        for t, m, s in zip(inputs, self.mean, self.std):
            t.sub_(m).div_(s)

        return inputs


class DeNormalize(object):
    """DeNormalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        result = inputs.clone()
        for i in range(result.size(0)):
            result[i, :, :] = result[i, :, :] * self.std[i] + self.mean[i]

        return result.mul_(255)


class ToTensor(object):
    """Convert a ``numpy.ndarray or Image`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        inputs (numpy.ndarray or Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    def __call__(self, inputs):
        if isinstance(inputs, Image.Image):
            channels = len(inputs.mode)
            inputs = np.array(inputs)
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], channels)
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))
        else:
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))

        return inputs.float()


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


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)

        return inputs
