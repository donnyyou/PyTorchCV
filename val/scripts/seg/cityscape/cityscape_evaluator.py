#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Evaluation of cityscape.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os
import argparse
from val.scripts.seg.cityscape.evaluation.evalPixelLevelSemanticLabelingObject import CArgs, EvalPixel
from utils.tools.configer import Configer



class CityScapeEvaluator(object):

    def evaluate(self, pred_dir, gt_dir):
        """
        :param pred_dir: directory of model output results(must be consistent with val directory)
        :param gt_dir: directory of  cityscape data(root)
        :return:
        """
        pred_path = pred_dir
        data_path = gt_dir
        print("evaluate the result...")
        args = CArgs(data_path=data_path, out_path=data_path, predict_path=pred_path)
        ob = EvalPixel(args)
        ob.run()


if __name__ == '__main__':
    evaluator = CityScapeEvaluator()
    data_path = "/dev/shm/DataSet/cityscapes/"
    res_path = "/dev/shm/DataSet/output/val/results/seg/cityscape/test_dir/image"
    evaluator.evaluate(res_path,data_path)



