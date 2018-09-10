#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Seg Model for semantic segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.seg.semantic.syncbn_deeplabv3 import SyncBNDeepLabV3
from models.seg.semantic.syncbn_pspnet import SyncBNPSPNet
from models.seg.semantic.syncbn_gridencnet import SyncBNGridEncNet
from models.seg.semantic.syncbn_embednet import SyncBNEmbedNet
from models.seg.semantic.syncbn_denseassp import SyncBNDenseASPP
from utils.tools.logger import Logger as Log


SEG_MODEL_DICT = {
    'syncbn_deeplabv3': SyncBNDeepLabV3,
    'syncbn_pspnet': SyncBNPSPNet,
    'syncbn_grid_encnet': SyncBNGridEncNet,
    'syncbn_embednet': SyncBNEmbedNet,
    'syncbn_denseaspp': SyncBNDenseASPP
}


class SegModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model
