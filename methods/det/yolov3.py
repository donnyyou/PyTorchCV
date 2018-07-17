#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Yolo v3.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datasets.det_data_loader import DetDataLoader
from loss.det_loss_manager import DetLossManager
from methods.tools.module_utilizer import ModuleUtilizer
from methods.tools.optim_scheduler import OptimScheduler
from models.det_model_manager import DetModelManager
from utils.layers.det.yolo_detection_layer import YOLODetectionLayer
from utils.helpers.det_helper import DetHelper
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from val.scripts.det.det_running_score import DetRunningScore
from vis.visualizer.det_visualizer import DetVisualizer


class YOLOv3(object):
    """
      The class for YOLO v3. Include train, val, test & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.det_visualizer = DetVisualizer(configer)
        self.det_loss_manager = DetLossManager(configer)
        self.det_model_manager = DetModelManager(configer)
        self.det_data_loader = DetDataLoader(configer)
        self.yolo_detection_layer = YOLODetectionLayer(configer)
        self.det_running_score = DetRunningScore(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.optim_scheduler = OptimScheduler(configer)

        self.det_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        self._init_model()

    def _init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = self.module_utilizer.load_net(self.det_net)

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(self._get_parameters())

        self.train_loader = self.det_data_loader.get_trainloader()
        self.val_loader = self.det_data_loader.get_valloader()

        self.det_loss = self.det_loss_manager.get_det_loss('yolov3_loss')

    def _get_parameters(self):

        return self.det_net.parameters()

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        if self.configer.get('network', 'resume') is not None and self.configer.get('iters') == 0:
            self.__val()

        self.det_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.configer.plus_one('epoch')
        self.scheduler.step(self.configer.get('epoch'))

        # data_tuple: (inputs, heatmap, maskmap, vecmap)
        for i, (inputs, bboxes, labels) in enumerate(self.train_loader):
            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs = self.module_utilizer.to_device(inputs)

            # Forward pass.
            output_list = self.det_net(inputs)

            # Compute the loss of the train batch & backward.
            loss = self.det_loss(output_list, bboxes, labels)

            self.train_losses.update(loss.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    self.configer.get('epoch'), self.configer.get('iters'),
                    self.configer.get('solver', 'display_iter'),
                    self.scheduler.get_lr(), batch_time=self.batch_time,
                    data_time=self.data_time, loss=self.train_losses))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # Check to val the current model.
            if self.val_loader is not None and \
               (self.configer.get('iters')) % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

    def __val(self):
        """
          Validation function during the train phase.
        """
        self.det_net.eval()
        start_time = time.time()
        with torch.no_grad():
            for j, (inputs, batch_gt_bboxes, batch_gt_labels) in enumerate(self.val_loader):
                # Forward pass.
                output_list = self.det_net(inputs)

                # Compute the loss of the val batch.
                loss = self.det_loss(output_list, batch_gt_bboxes, batch_gt_labels)
                self.val_losses.update(loss.item(), inputs.size(0))

                batch_detections = self.__decode(output_list)
                batch_pred_bboxes = self.__get_object_list(batch_detections)

                self.det_running_score.update(batch_pred_bboxes, batch_gt_bboxes, batch_gt_labels)

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.module_utilizer.save_net(self.det_net, metric='iters')
            # Print the log info & reset the states.
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            Log.info('Val mAP: {}'.format(self.det_running_score.get_mAP()))
            self.det_running_score.reset()
            self.batch_time.reset()
            self.val_losses.reset()
            self.det_net.train()

    def __decode(self, output_list):
        """Transform predicted loc/conf back to real bbox locations and class labels.

        Args:
          loc: (tensor) predicted loc, sized [8732, 4].
          conf: (tensor) predicted conf, sized [8732, 21].

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].

        """
        anchors_list = self.configer.get('gt', 'anchors')
        assert len(anchors_list) == len(output_list)

        pred_list = list()
        for outputs, anchors in zip(output_list, anchors_list):
            pred_list.append(self.yolo_detection_layer(outputs, anchors, is_training=False))

        batch_pred_bboxes = torch.cat(pred_list, 1)

        batch_detections = self.__nms(batch_pred_bboxes)
        return batch_detections

    def __nms(self, prediction):
        """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
        """

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            # Filter out confidence scores below threshold
            conf_mask = (image_pred[:, 4] > self.configer.get('vis', 'obj_threshold')).squeeze()
            image_pred = image_pred[conf_mask]
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue

            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5:5 + self.configer.get('data', 'num_classes')], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            # Iterate through all predicted classes
            unique_labels = detections[:, -1].cpu().unique()
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
            for c in unique_labels:
                # Get the detections with the particular class
                detections_class = detections[detections[:, -1] == c]
                # Sort the detections by maximum objectness confidence
                _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
                detections_class = detections_class[conf_sort_index]
                # Perform non-maximum suppression
                max_detections = []
                while detections_class.size(0):
                    # Get detection with highest confidence and save as max detection
                    max_detections.append(detections_class[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(detections_class) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = DetHelper.bbox_iou(torch.from_numpy(np.array(max_detections[-1])),
                                              torch.from_numpy(np.array(detections_class[1:])))
                    # Remove detections with IoU >= NMS threshold
                    detections_class = detections_class[1:][ious[0] < self.configer.get('nms', 'overlap_threshold')]

                max_detections = torch.cat(max_detections).data
                # Add max detections to outputs
                output[image_i] = max_detections if output[image_i] is None else torch.cat(
                    (output[image_i], max_detections))

        return output

    def __get_object_list(self, batch_detections):
        batch_pred_bboxes = list()
        for idx, detections in enumerate(batch_detections):
            object_list = list()
            if detections is not None:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    xmin = x1.cpu().item()
                    ymin = y1.cpu().item()
                    xmax = x2.cpu().item()
                    ymax = y2.cpu().item()
                    cf = conf.cpu().item()
                    cls_pred = cls_pred.cpu().item()
                    object_list.append([xmin, ymin, xmax, ymax, int(cls_pred), float('%.2f' % cf)])

            batch_pred_bboxes.append(object_list)

        return batch_pred_bboxes

    def train(self):
        cudnn.benchmark = True
        while self.configer.get('epoch') < self.configer.get('solver', 'max_epoch'):
            self.__train()
            if self.configer.get('epoch') == self.configer.get('solver', 'max_epoch'):
                break


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
