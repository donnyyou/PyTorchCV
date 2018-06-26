#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualizer for segmentation.


import os
import cv2
import numpy as np

from datasets.tools.transforms import DeNormalize
from utils.tools.logger import Logger as Log


SEG_DIR = 'vis/results/seg'


class SegVisualizer(object):

    def __init__(self, configer):
        self.configer = configer
