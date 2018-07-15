#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss Manager for Object Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loss.modules.det_modules import SSDFocalLoss, SSDMultiBoxLoss
from utils.tools.logger import Logger as Log


DET_LOSS_DICT = {
    'focal_loss': SSDFocalLoss,
    'multibox_loss': SSDMultiBoxLoss,
}


class DetLossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def get_det_loss(self, key):
        if key not in DET_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = DET_LOSS_DICT[key](self.configer)

        return loss
