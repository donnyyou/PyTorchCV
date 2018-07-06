#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from extensions.layers.encoding.syncbn import BatchNorm2d as sync_bn


class BatchNorm2d(object):

    def __call__(self, inchannels, sync=False):
        if not sync:
            return nn.BatchNorm2d(inchannels)

        else:
            return sync_bn(inchannels)

