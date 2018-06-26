#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Test class for associative embedding.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from methods.tools.module_utilizer import ModuleUtilizer
from models.pose_model_manager import PoseModelManager
from vis.visualizer.pose_visualizer import PoseVisualizer


class AssociativeEmbeddingTest(object):
    def __init__(self, configer):
        self.configer = configer

        self.pose_vis = PoseVisualizer(configer)
        self.pose_model_manager = PoseModelManager(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.pose_net = None