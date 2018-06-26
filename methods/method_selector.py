#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from methods.cls.fc_classifier import FCClassifier
from methods.cls.fc_classifier_test import FCClassifierTest
from methods.det.single_shot_detector import SingleShotDetector
from methods.det.single_shot_detector_test import SingleShotDetectorTest
from methods.pose.associative_embedding import AssociativeEmbedding
from methods.pose.associative_embedding_test import AssociativeEmbeddingTest
from methods.pose.conv_pose_machine import ConvPoseMachine
from methods.pose.conv_pose_machine_test import ConvPoseMachineTest
from methods.pose.open_pose import OpenPose
from methods.pose.open_pose_test import OpenPoseTest
from methods.pose.rpn_pose import RPNPose
from methods.pose.rpn_pose_test import RPNPoseTest
from methods.pose.capsule_pose import CapsulePose
from methods.pose.capsule_pose_test import CapsulePoseTest
from methods.seg.fcn_segmentor import FCNSegmentor
from methods.seg.fcn_segmentor_test import FCNSegmentorTest
from utils.tools.logger import Logger as Log


POSE_METHOD_DICT = {
    'open_pose': OpenPose,
    'conv_pose_machine': ConvPoseMachine,
    'associative_embedding': AssociativeEmbedding,
    'rpn_pose': RPNPose,
    'capsule_pose': CapsulePose,
}
POSE_TEST_DICT = {
    'open_pose': OpenPoseTest,
    'conv_pose_machine': ConvPoseMachineTest,
    'associative_embedding': AssociativeEmbeddingTest,
    'rpn_pose': RPNPoseTest,
    'capsule_pose': CapsulePoseTest,
}

SEG_METHOD_DICT = {
    'fcn_segmentor': FCNSegmentor,
}
SEG_TEST_DICT = {
    'fcn_segmentor': FCNSegmentorTest,
}

DET_METHOD_DICT = {
    'single_shot_detector': SingleShotDetector,
}
DET_TEST_DICT = {
    'single_shot_detector': SingleShotDetectorTest,
}

CLS_METHOD_DICT = {
    'fc_classifier': FCClassifier,
}
CLS_TEST_DICT = {
    'fc_classifier': FCClassifierTest,
}

MULTITASK_METHOD_DICT = {

}
MULTITASK_TEST_DICT = {

}


class MethodSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def select_pose_method(self):
        key = self.configer.get('method')
        if key not in POSE_METHOD_DICT or key not in POSE_TEST_DICT:
            Log.error('Pose Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return POSE_METHOD_DICT[key](self.configer)
        else:
            return POSE_TEST_DICT[key](self.configer)

    def select_det_method(self):
        key = self.configer.get('method')
        if key not in DET_METHOD_DICT or key not in DET_TEST_DICT:
            Log.error('Det Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return DET_METHOD_DICT[key](self.configer)
        else:
            return DET_TEST_DICT[key](self.configer)

    def select_seg_method(self):
        key = self.configer.get('method')
        if key not in SEG_METHOD_DICT or key not in SEG_TEST_DICT:
            Log.error('Det Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return SEG_METHOD_DICT[key](self.configer)
        else:
            return SEG_TEST_DICT[key](self.configer)

    def select_cls_method(self):
        key = self.configer.get('method')
        if key not in CLS_METHOD_DICT or key not in CLS_TEST_DICT:
            Log.error('Cls Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return CLS_METHOD_DICT[key](self.configer)
        else:
            return CLS_TEST_DICT[key](self.configer)

    def select_multitask_method(self):
        key = self.configer.get('method')
        if key not in MULTITASK_METHOD_DICT or key not in MULTITASK_TEST_DICT:
            Log.error('Multitask Method: {} is not valid.'.format(key))
            exit(1)

        if self.configer.get('phase') == 'train':
            return MULTITASK_METHOD_DICT[key](self.configer)
        else:
            return MULTITASK_TEST_DICT[key](self.configer)
