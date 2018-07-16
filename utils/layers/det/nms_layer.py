#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class NMSLayer(object):

    def __init__(self, configer):
        self.configer = configer

    def __call__(self, *args, **kwargs):
        return None