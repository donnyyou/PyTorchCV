#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from methods.tools.data_transformer import DataTransformer

DATA_KEYS = ['img', 'label', 'labelmap', 'maskmap', 'kpts', 'bboxes', 'labels', 'polygons']


class CollateFunctions(object):

    @staticmethod
    def our_collate(batch, data_keys=None, configer=None, trans_dict=None):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).
        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations
        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on 0 dim
        """

        transposed = [list(sample) for sample in zip(*batch)]
        new_transposed = []
        index = 0
        for key in DATA_KEYS:
            if key in data_keys:
                new_transposed.append(transposed[index])
                index += 1
            else:
                new_transposed.append(None)

        new_transposed.append(trans_dict)
        data_dict = DataTransformer(configer)(*new_transposed)
        print(data_dict)
        return data_dict

    @staticmethod
    def _default_collate(batch):
        transposed = [list(sample) for sample in zip(*batch)]
        return []