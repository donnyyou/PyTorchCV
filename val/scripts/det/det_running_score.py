#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Object Detection running score.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class DetRunningScore(object):
    def __init__(self, configer):
        self.configer = configer

    def update(self):
        pass

    def get_mAP(self):
        pass

    def reset(self):
        pass