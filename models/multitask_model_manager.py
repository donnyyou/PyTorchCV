#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Multitask Model for computer vision.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.tools.logger import Logger as Log


MULTITASK_MODEL_DICT = {
}


class DetModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def object_detector(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in MULTITASK_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = MULTITASK_MODEL_DICT[model_name](self.configer)

        return model