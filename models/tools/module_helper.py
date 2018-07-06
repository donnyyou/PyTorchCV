#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from extensions.layers.encoding.syncbn import BatchNorm2d


class ModuleHelper(object):

    def __init__(self, configer):
        self.configer = configer

    def bn(self):
        if not self.configer.is_empty('network', 'syncbn') and self.configer.get('network', 'syncbn'):
            return BatchNorm2d

        else:
            return nn.BatchNorm2d
        