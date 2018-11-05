#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from utils.tools.logger import Logger as Log


class ModuleHelper(object):

    @staticmethod
    def BatchNorm2d(bn_type='torch'):
        if bn_type == 'torchbn':
            return nn.BatchNorm2d

        elif bn_type == 'syncbn':
            from extensions.layers.syncbn.module import BatchNorm2d
            return BatchNorm2d

        else:
            Log.error('Not support BN type: {}.'.format(bn_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True):
        if pretrained is None:
            return model

        if all_match:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model.load_state_dict(pretrained_dict)

        else:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = model.state_dict()
            load_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
            Log.info('Matched Keys: {}'.format(load_dict.keys()))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join('~', '.PyTorchCV', 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            Log.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        Log.info('Loading pretrained model:{}'.format(cached_file))
        return torch.load(cached_file, map_location=map_location)

