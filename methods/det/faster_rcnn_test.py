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
import torch.nn.functional as F
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
from utils.layers.det.fr_priorbox_layer import FRPriorBoxLayer
from utils.layers.det.fr_roi_generator import FRRoiGenerator
from utils.tools.logger import Logger as Log
from vis.parser.det_parser import DetParser
from vis.visualizer.det_visualizer import DetVisualizer


class FastRCNNTest(object):
    def __init__(self, configer):
        self.configer = configer

        self.det_visualizer = DetVisualizer(configer)
        self.det_parser = DetParser(configer)
        self.det_model_manager = DetModelManager(configer)
        self.det_data_loader = DetDataLoader(configer)
        self.det_data_utilizer = DetDataUtilizer(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.fr_priorbox_layer = FRPriorBoxLayer(configer)
        self.fr_roi_generator = FRRoiGenerator(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.det_net = None

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = self.module_utilizer.load_net(self.det_net)
        self.module_utilizer.set_status(self.det_net, status='test')

    def __test_img(self, image_path, json_path, raw_path, vis_path):
        Log.info('Image Path: {}'.format(image_path))
        ori_img_rgb = ImageHelper.img2np(ImageHelper.pil_open_rgb(image_path))
        ori_img_bgr = ImageHelper.rgb2bgr(ori_img_rgb)
        inputs = ImageHelper.resize(ori_img_rgb, tuple(self.configer.get('data', 'input_size')), Image.CUBIC)
        inputs = ToTensor()(inputs)
        inputs = Normalize(mean=self.configer.get('trans_params', 'mean'),
                           std=self.configer.get('trans_params', 'std'))(inputs)

        with torch.no_grad():
            inputs = inputs.unsqueeze(0).to(self.device)
            # Forward pass.
            feat = self.det_net.extractor(inputs)
            rpn_locs, rpn_scores = self.det_net.rpn(inputs)

            test_indices_and_rois = self.fr_roi_generator(rpn_locs, rpn_scores,
                                                          self.configer.get('rpn', 'n_test_pre_nms'),
                                                          self.configer.get('rpn', 'n_test_post_nms'))
            test_roi_locs, test_roi_scores = self.det_net.roi_head(feat, test_indices_and_rois)

        batch_detections = self.decode(test_roi_locs,
                                       test_roi_scores,
                                       test_indices_and_rois,
                                       self.configer,
                                       inputs.size(0))
        json_dict = self.__get_info_tree(batch_detections[0], ori_img_rgb)

        image_canvas = self.det_parser.draw_bboxes(ori_img_bgr.copy(),
                                                   json_dict,
                                                   conf_threshold=self.configer.get('vis', 'conf_threshold'))
        cv2.imwrite(vis_path, image_canvas)
        cv2.imwrite(raw_path, ori_img_bgr)

        Log.info('Json Path: {}'.format(json_path))
        JsonHelper.save_file(json_dict, json_path)
        return json_dict

    @staticmethod
    def decode(roi_locs, roi_scores, indices_and_rois, configer, batch_size):
        num_classes = configer.get('data', 'num_classes')
        mean = torch.Tensor(configer.get('roi', 'loc_normalize_mean')).repeat(num_classes)
        std = torch.Tensor(configer.get('roi', 'loc_normalize_std')).repeat(num_classes)

        if roi_locs.is_cuda:
            mean = mean.cuda()
            std = std.cuda()

        roi_locs = (roi_locs * std + mean)
        roi_locs = roi_locs.contiguous().view(-1, num_classes, 4)

        rois = indices_and_rois[:, 1:]
        rois = rois.contiguous().view(-1, 1, 4).expand_as(roi_locs)
        wh = roi_locs[:, :, 2:] * (rois[:, :, 2:] - rois[:, :, :2])
        cxcy = roi_locs[:, :, :2] * (rois[:, :, 2:] - rois[:, :, :2]) + (rois[:, :, :2] + rois[:, :, 2:]) / 2
        dst_bbox = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 2)  # [b, 8732,4]

        # clip bounding box
        dst_bbox[:, :, 0::2] = (dst_bbox[:, :, 0::2]).clamp(min=0, max=configer.get('data', 'input_size')[0])
        dst_bbox[:, :, 1::2] = (dst_bbox[:, :, 1::2]).clamp(min=0, max=configer.get('data', 'input_size')[1])

        cls_prob = F.softmax(roi_scores, dim=1)
        cls_label = torch.LongTensor([i for i in range(num_classes)])\
            .contiguous().view(1, num_classes).repeat(indices_and_rois.size(0), 1)

        print(roi_scores.size())
        print(cls_label.size())
        print(dst_bbox.size())

        output = [None for _ in range(batch_size)]
        for i in range(batch_size):
            batch_index = (indices_and_rois[:, 0] == i).nonzero().contiguous().view(-1,)
            tmp_dst_bbox = dst_bbox[batch_index]
            tmp_cls_prob = cls_prob[batch_index]
            tmp_cls_label = cls_label[batch_index]

            mask = tmp_cls_prob > configer.get('vis', 'conf_threshold')

            tmp_dst_bbox = tmp_dst_bbox[mask].contiguous().view(-1, 4)
            if tmp_dst_bbox.numel() == 0:
                continue

            tmp_cls_prob = tmp_cls_prob[mask].contiguous().view(-1,).unsqueeze(1)
            tmp_cls_label = tmp_cls_label[mask].contiguous().view(-1,).unsqueeze(1)
            valid_preds = torch.cat((tmp_dst_bbox, tmp_cls_prob.float(), tmp_cls_label.float()), 1)

            keep = DetHelper.cls_nms(valid_preds[:, :4],
                                     scores=valid_preds[:, 4],
                                     labels=valid_preds[:, 5],
                                     nms_threshold=configer.get('nms', 'overlap_threshold'),
                                     mode=configer.get('nms', 'mode'))

            output[i] = valid_preds[keep]

        return output

    def __get_info_tree(self, detections, image_raw):
        height, width, _ = image_raw.shape
        json_dict = dict()
        object_list = list()
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_pred in detections:
                object_dict = dict()
                xmin = x1.cpu().item() * width
                ymin = y1.cpu().item() * height
                xmax = x2.cpu().item() * width
                ymax = y2.cpu().item() * height
                object_dict['bbox'] = [xmin, ymin, xmax, ymax]
                object_dict['label'] = int(cls_pred.cpu().item()) - 1
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

        self.module_utilizer.set_status(self.det_net, status='debug')
        val_data_loader = self.det_data_loader.get_valloader()
        count = 0
        for i, (inputs, batch_gt_bboxes, batch_gt_labels) in enumerate(val_data_loader):
            gt_rpn_locs, gt_rpn_labels = self.det_data_utilizer.rpn_batch_encode(batch_gt_bboxes,
                                                                                 self.fr_priorbox_layer())
            eye_matrix = torch.eye(2)
            gt_rpn_scores = eye_matrix[gt_rpn_labels.view(-1)].view(inputs.size(0), -1, 2)
            test_indices_and_rois = self.fr_roi_generator(gt_rpn_locs, gt_rpn_scores,
                                                          self.configer.get('rpn', 'n_test_pre_nms'),
                                                          self.configer.get('rpn', 'n_test_post_nms'))
            sample_rois, gt_roi_locs, gt_roi_labels = self.det_data_utilizer.roi_batch_encode(
                batch_gt_bboxes, batch_gt_labels, indices_and_rois=test_indices_and_rois)
            eye_matrix = torch.eye(self.configer.get('data', 'num_classes'))
            gt_cls_roi_locs = torch.zeros_like(gt_roi_locs).repeat(1, self.configer.get('data', 'num_classes'))

            gt_roi_scores = eye_matrix[gt_roi_labels.view(-1)].view(gt_roi_labels.size(0),
                                                                    self.configer.get('data', 'num_classes'))
            batch_detections = FastRCNNTest.decode(gt_cls_roi_locs, gt_roi_scores,
                                                   sample_rois, self.configer, inputs.size(0))

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
                                                           conf_threshold=self.configer.get('vis', 'conf_threshold'))

                cv2.imwrite(os.path.join(base_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()