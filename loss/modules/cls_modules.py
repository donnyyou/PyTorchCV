#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, configer=None):
        super(CrossEntropyLoss, self).__init__()
        self.configer = configer

        weight = None
        if not self.configer.is_empty('cross_entropy_loss', 'weight'):
            weight = self.configer.get('cross_entropy_loss', 'weight')

        size_average = self.configer.get('cross_entropy_loss', 'size_average')
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average)

    def forward(self, inputs, targets):
        return self.cross_entropy_loss(inputs, targets)
