#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.backbones.vgg.vgg_backbone import VGGBackbone
from models.backbones.resnet.resnet_backbone import ResNetBackbone
from models.backbones.mobilenet.mobilenet_backbone import MobileNetBackbone
from models.backbones.densenet.densenet_backbone import DenseNetBackbone
from models.backbones.squeezenet.squeezenet_backbone import SqueezeNetBackbone
from utils.tools.logger import Logger as Log


class BackboneSelector(object):

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self):
        backbone = self.configer.get('network', 'backbone')

        model = None
        if 'vgg' in backbone:
            model = VGGBackbone(self.configer)()

        elif 'resnet' in backbone:
            model = ResNetBackbone(self.configer)()

        elif 'mobilenet' in backbone:
            model = MobileNetBackbone(self.configer)()

        elif 'densenet' in backbone:
            model = DenseNetBackbone(self.configer)()

        elif 'squeezenet' in backbone:
            model = SqueezeNetBackbone(self.configer)

        else:
            Log.error('Backbone {} is invalid.'.format(backbone))
            exit(1)

        return model