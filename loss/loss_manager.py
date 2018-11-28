#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss Manager for Image Classification.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loss.modules.cls_modules import FCClsLoss
from loss.modules.det_modules import FRDetLoss
from loss.modules.det_modules import SSDMultiBoxLoss
from loss.modules.det_modules import YOLOv3DetLoss
from loss.modules.pose_modules import OPPoseLoss
from loss.modules.seg_modules import FCNSegLoss
from utils.tools.logger import Logger as Log


CLS_LOSS_DICT = {
    'fc_cls_loss': FCClsLoss,
}

DET_LOSS_DICT = {
    'ssd_det_loss': SSDMultiBoxLoss,
    'yolov3_det_loss': YOLOv3DetLoss,
    'fr_det_loss': FRDetLoss
}

POSE_LOSS_DICT = {
    'op_poss_loss': OPPoseLoss,
}

SEG_LOSS_DICT = {
    'fcn_seg_loss': FCNSegLoss
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _parallel(self, loss):
        if self.configer.get('network', 'loss_balance') and len(self.configer.get('gpu')) > 1:
            from extensions.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss

    def get_cls_loss(self, key):
        if key not in CLS_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = CLS_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

    def get_seg_loss(self, key):
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

    def get_det_loss(self, key):
        if key not in DET_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = DET_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

    def get_pose_loss(self, key):
        if key not in POSE_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = POSE_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

