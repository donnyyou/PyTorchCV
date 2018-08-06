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
from PIL import Image

from datasets.det_data_loader import DetDataLoader
from datasets.det.det_data_utilizer import DetDataUtilizer
from datasets.tools.transforms import Normalize, ToTensor, DeNormalize
from methods.tools.module_utilizer import ModuleUtilizer
from models.det_model_manager import DetModelManager
from utils.helpers.det_helper import DetHelper
from utils.helpers.image_helper import ImageHelper
from utils.helpers.file_helper import FileHelper
from utils.helpers.json_helper import JsonHelper
from utils.layers.det.yolo_detection_layer import YOLODetectionLayer
from utils.tools.logger import Logger as Log
from vis.parser.det_parser import DetParser
from vis.visualizer.det_visualizer import DetVisualizer


class YOLOv3Test(object):
    def __init__(self, configer):
        self.configer = configer

        self.det_visualizer = DetVisualizer(configer)
        self.det_parser = DetParser(configer)
        self.det_model_manager = DetModelManager(configer)
        self.det_data_loader = DetDataLoader(configer)
        self.det_data_utilizer = DetDataUtilizer(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.yolo_detection_layer = YOLODetectionLayer(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.det_net = None

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = self.module_utilizer.load_net(self.det_net)
        self.module_utilizer.set_status(self.det_net, status='test')

    def __test_img(self, image_path, json_path, raw_path, vis_path):
        Log.info('Image Path: {}'.format(image_path))
        ori_img_rgb = ImageHelper.img2np(ImageHelper.pil_read_image(image_path))
        ori_img_bgr = ImageHelper.rgb2bgr(ori_img_rgb)
        inputs = ImageHelper.resize(ori_img_rgb, tuple(self.configer.get('data', 'input_size')), Image.CUBIC)
        inputs = ToTensor()(inputs)
        inputs = Normalize(mean=self.configer.get('trans_params', 'mean'),
                           std=self.configer.get('trans_params', 'std'))(inputs)

        with torch.no_grad():
            inputs = inputs.unsqueeze(0).to(self.device)
            _, detections = self.det_net(inputs)

        batch_detections = self.decode(detections, self.configer)
        json_dict = self.__get_info_tree(batch_detections[0], ori_img_rgb)[0]

        image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                   json_dict,
                                                   conf_threshold=self.configer.get('vis', 'conf_threshold'))
        cv2.imwrite(vis_path, image_canvas)
        cv2.imwrite(raw_path, ori_img_bgr)

        Log.info('Json Path: {}'.format(json_path))
        JsonHelper.save_file(json_dict, json_path)
        return json_dict

    @staticmethod
    def decode(batch_pred_bboxes, configer):
        """Transform predicted loc/conf back to real bbox locations and class labels.

        Args:
          loc: (tensor) predicted loc, sized [8732, 4].
          conf: (tensor) predicted conf, sized [8732, 21].

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].

        """
        box_corner = batch_pred_bboxes.new(batch_pred_bboxes.shape)
        box_corner[:, :, 0] = batch_pred_bboxes[:, :, 0] - batch_pred_bboxes[:, :, 2] / 2
        box_corner[:, :, 1] = batch_pred_bboxes[:, :, 1] - batch_pred_bboxes[:, :, 3] / 2
        box_corner[:, :, 2] = batch_pred_bboxes[:, :, 0] + batch_pred_bboxes[:, :, 2] / 2
        box_corner[:, :, 3] = batch_pred_bboxes[:, :, 1] + batch_pred_bboxes[:, :, 3] / 2
        batch_pred_bboxes[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(batch_pred_bboxes))]
        for image_i, image_pred in enumerate(batch_pred_bboxes):
            # Filter out confidence scores below threshold
            conf_mask = (image_pred[:, 4] > configer.get('vis', 'obj_threshold')).squeeze()
            image_pred = image_pred[conf_mask]
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue

            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5:5 + configer.get('data', 'num_classes')], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            keep_index = DetHelper.cls_nms(image_pred[:, :4],
                                           scores=image_pred[:, 4],
                                           labels=class_pred.squeeze(1),
                                           nms_threshold=configer.get('nms', 'overlap_threshold'),
                                           mode=configer.get('nms', 'mode'))

            output[image_i] = detections[keep_index]

        return output

    def __get_info_tree(self, detections, image_raw):
        height, width, _ = image_raw.shape
        json_dict = dict()
        object_list = list()
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                object_dict = dict()
                xmin = x1.cpu().item() * width
                ymin = y1.cpu().item() * height
                xmax = x2.cpu().item() * width
                ymax = y2.cpu().item() * height
                object_dict['bbox'] = [xmin, ymin, xmax, ymax]
                object_dict['label'] = int(cls_pred.cpu().item())
                object_dict['score'] = float('%.2f' % conf.cpu().item())

                object_list.append(object_dict)

        json_dict['objects'] = object_list

        return json_dict

    def test(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/det', self.configer.get('dataset'))

        test_img = self.configer.get('test_img')
        test_dir = self.configer.get('test_dir')
        if test_img is None and test_dir is None:
            Log.error('test_img & test_dir not exists.')
            exit(1)

        if test_img is not None and test_dir is not None:
            Log.error('Either test_img or test_dir.')
            exit(1)

        if test_img is not None:
            base_dir = os.path.join(base_dir, 'test_img')
            filename = test_img.rstrip().split('/')[-1]
            json_path = os.path.join(base_dir, 'json', '{}.json'.format('.'.join(filename.split('.')[:-1])))
            raw_path = os.path.join(base_dir, 'raw', filename)
            vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
            if not os.path.exists(os.path.dirname(json_path)):
                os.makedirs(os.path.dirname(json_path))

            if not os.path.exists(os.path.dirname(raw_path)):
                os.makedirs(os.path.dirname(raw_path))

            if not os.path.exists(os.path.dirname(vis_path)):
                os.makedirs(os.path.dirname(vis_path))

            self.__test_img(test_img, json_path, raw_path, vis_path)

        else:
            base_dir = os.path.join(base_dir, 'test_dir', test_dir.rstrip('/').split('/')[-1])
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            for filename in FileHelper.list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                json_path = os.path.join(base_dir, 'json', '{}.json'.format('.'.join(filename.split('.')[:-1])))
                raw_path = os.path.join(base_dir, 'raw', filename)
                vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
                if not os.path.exists(os.path.dirname(json_path)):
                    os.makedirs(os.path.dirname(json_path))

                if not os.path.exists(os.path.dirname(raw_path)):
                    os.makedirs(os.path.dirname(raw_path))

                if not os.path.exists(os.path.dirname(vis_path)):
                    os.makedirs(os.path.dirname(vis_path))

                self.__test_img(image_path, json_path, raw_path, vis_path)

    def debug(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/det', self.configer.get('dataset'), 'debug')

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        val_data_loader = self.det_data_loader.get_valloader()

        count = 0
        self.module_utilizer.set_status(self.det_net, status='debug')
        input_size = self.configer.get('data', 'input_size')
        for i, (inputs, bboxes, labels) in enumerate(val_data_loader):
            targets, _, _ = self.det_data_utilizer.yolo_batch_encode(bboxes, labels)
            targets = targets.to(self.device)
            anchors_list = self.configer.get('gt', 'anchors_list')
            output_list = list()
            be_c = 0
            for f_index, anchors in enumerate(anchors_list):
                feat_stride = self.configer.get('network', 'stride_list')[f_index]
                fm_size = [int(round(border / feat_stride)) for border in input_size]
                num_c = len(anchors) * fm_size[0] * fm_size[1]
                output_list.append(targets[:, be_c:be_c+num_c].contiguous()
                                   .view(targets.size(0), len(anchors), fm_size[1], fm_size[0], -1)
                                   .permute(0, 1, 4, 2, 3).contiguous()
                                   .view(targets.size(0), -1, fm_size[1], fm_size[0]))

                be_c += num_c

            batch_detections = self.decode(self.yolo_detection_layer(output_list)[1], self.configer)

            for j in range(inputs.size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                ori_img_rgb = DeNormalize(mean=self.configer.get('trans_params', 'mean'),
                                          std=self.configer.get('trans_params', 'std'))(inputs[j])
                ori_img_rgb = ori_img_rgb.numpy().transpose(1, 2, 0).astype(np.uint8)
                ori_img_bgr = cv2.cvtColor(ori_img_rgb, cv2.COLOR_RGB2BGR)

                json_dict = self.__get_info_tree(batch_detections[j], ori_img_rgb)

                image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                           json_dict,
                                                           conf_threshold=self.configer.get('vis', 'obj_threshold'))

                cv2.imwrite(os.path.join(base_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()
