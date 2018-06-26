#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# VGG models.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.backbones.vgg.vgg_models import VGGModels


class VGGBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.vgg_models = VGGModels(self.configer)

    def __call__(self):
        arch = self.configer.get('network', 'backbone')
        if 'bn' in arch:
            arch_net = self.vgg_models.vgg_bn()

        else:
            arch_net = self.vgg_models.vgg()

        return arch_net