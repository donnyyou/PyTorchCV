#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch.nn import functional as F
from torch import nn


class ROIPoolingLayer(nn.Module):

    def __init__(self, configer):
        self.configer = configer
