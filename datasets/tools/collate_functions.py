#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.tools.data_transformer import DataTransformer


DATA_KEYS_SEQ = ['img', 'label', 'imgscale', 'labelmap', 'maskmap', 'kpts', 'bboxes', 'labels', 'polygons']


class CollateFunctions(object):

    @staticmethod
    def our_collate(batch, data_keys=None, configer=None, trans_dict=None):
        transposed = [list(sample) for sample in zip(*batch)]
        new_transposed = []
        index = 0
        for key in DATA_KEYS_SEQ:
            if key in data_keys:
                new_transposed.append(transposed[index])
                index += 1
            else:
                new_transposed.append(None)

        new_transposed.append(trans_dict)
        data_dict = DataTransformer(configer)(*new_transposed)
        return data_dict

    @staticmethod
    def _default_collate(batch, configer=None,):
        transposed = [list(sample) for sample in zip(*batch)]
        return [DataTransformer(configer).stack(item, 0) for item in transposed]