#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from datasets.tools.transforms import Normalize, ToTensor
from models.det_model_manager import DetModelManager
from utils.layers.det.priorbox_layer import SSDPriorBoxLayer
from utils.tools.configer import Configer
from utils.tools.logger import Logger as Log
from vis.visualizer.det_visualizer import DetVisualizer


class SingleShotDetectorDeploy(object):
    def __init__(self, model_path=None, gpu_id=0):
        self.model_path = model_path
        self.det_visualizer = DetVisualizer(self.configer)
        self.det_model_manager = DetModelManager(self.configer)
        self.default_boxes = SSDPriorBoxLayer(self.configer)()
        self.det_net = None

        self._init_model(model_path=model_path, gpu_id=gpu_id)

    def _init_model(self, model_path, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = nn.DataParallel(self.det_net).cuda()
        if model_path is not None:
            model_dict = torch.load(model_path)
            self.det_net.load_state_dict(model_dict['state_dict'])
            self.configer = Configer(config_dict=model_dict['config_dict'])
        else:
            Log.error('Model Path is not existed.')
            exit(1)

        self.det_net.eval()

    def inference(self, image_rgb):
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        inputs = cv2.resize(image_rgb, tuple(self.configer.get('data', 'input_size')))
        inputs = ToTensor()(inputs)
        inputs = Normalize(mean=self.configer.get('trans_params', 'mean'),
                           std=self.configer.get('trans_params', 'std'))(inputs)

        inputs = Variable(inputs.unsqueeze(0).cuda(), volatile=True)
        bbox, cls = self.det_net(inputs)
        bbox = bbox.cpu().data.squeeze(0)
        cls = F.softmax(cls.cpu().squeeze(0), dim=-1).data
        boxes, lbls, scores, has_obj = self.__decode(bbox, cls)
        if has_obj:
            boxes = boxes.cpu().numpy()
            boxes = np.clip(boxes, 0, 1)
            lbls = lbls.cpu().numpy()
            scores = scores.cpu().numpy()
            img_shape = image_bgr.shape
            for i in range(len(boxes)):
                boxes[i][0] = int(boxes[i][0] * img_shape[1])
                boxes[i][2] = int(boxes[i][2] * img_shape[1])
                boxes[i][1] = int(boxes[i][1] * img_shape[0])
                boxes[i][3] = int(boxes[i][3] * img_shape[0])

            img_canvas = self.__draw_box(image_bgr, boxes, lbls, scores)

            # if is_save_txt:
            #    self.__save_txt(save_path, boxes, lbls, scores, img_size)
        else:
            # print('None obj detected!')
            img_canvas = image_bgr

        # Boxes is within 0-1.
        return img_canvas, lbls, scores, boxes, has_obj

    def __draw_box(self, img_raw, box_list, label_list, conf):
        img_canvas = img_raw.copy()

        for bbox, label, cf in zip(box_list, label_list, conf):
            if cf < self.configer.get('vis', 'conf_threshold'):
                continue

            class_name = self.configer.get('details', 'name_seq')[label - 1] + str(cf)
            c = self.configer.get('details', 'color_list')[label - 1]
            cv2.rectangle(img_canvas, (max(0, int(bbox[0]-10)), max(0, int(bbox[1]-10))),
                          (min(img_canvas.shape[1], int(bbox[2]+10)),
                           min(img_canvas.shape[0], int(bbox[3]+10))), color=c, thickness=3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_canvas, class_name, (int(bbox[0]+5), int(bbox[3]-5)), font, fontScale=0.5, color=c, thickness=2)

        return img_canvas

    def __nms(self, bboxes, scores, mode='union'):
        """Non maximum suppression.

        Args:
          bboxes(tensor): bounding boxes, sized [N,4].
          scores(tensor): bbox scores, sized [N,].
          threshold(float): overlap threshold.
          mode(str): 'union' or 'min'.

        Returns:
          keep(tensor): selected indices.

        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        """

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            if self.configer.get('nms', 'mode') == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif self.configer.get('nms', 'mode') == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= self.configer.get('nms', 'overlap_threshold')).nonzero().squeeze()
            if ids.numel() == 0:
                break

            order = order[ids + 1]

        return torch.LongTensor(keep)

    def __decode(self, loc, conf):
        """Transform predicted loc/conf back to real bbox locations and class labels.

        Args:
          loc: (tensor) predicted loc, sized [8732, 4].
          conf: (tensor) predicted conf, sized [8732, 21].

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].

        """
        has_obj = False
        variances = [0.1, 0.2]
        wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:]
        cxcy = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)  # [8732,4]

        max_conf, labels = conf.max(1)  # [8732,1]
        ids = labels.nonzero()
        tmp = ids.cpu().numpy()

        if tmp.__len__() > 0:
            # print('detected %d objs' % tmp.__len__())
            ids = ids.squeeze(1)  # [#boxes,]
            has_obj = True
        else:
            print('None obj detected!')
            return 0, 0, 0, has_obj

        keep = self.__nms(boxes[ids], max_conf[ids])
        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep], has_obj

