#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from datasets.det_data_loader import DetDataLoader
from datasets.det.det_data_utilizer import DetDataUtilizer
from loss.det_loss_manager import DetLossManager
from methods.tools.module_utilizer import ModuleUtilizer
from methods.tools.optim_scheduler import OptimScheduler
from models.det_model_manager import DetModelManager
from utils.layers.det.ssd_priorbox_layer import SSDPriorBoxLayer
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from val.scripts.det.det_running_score import DetRunningScore
from vis.visualizer.det_visualizer import DetVisualizer


class SingleShotDetector(object):
    """
      The class for Single Shot Detector. Include train, val, test & predict.
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
        self.det_data_utilizer = DetDataUtilizer(configer)
        self.ssd_priorbox_layer = SSDPriorBoxLayer(configer)
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

        self.det_loss = self.det_loss_manager.get_det_loss('ssd_multibox_loss')

    def _get_parameters(self):

        return self.det_net.parameters()

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        if self.configer.get('network', 'resume') is not None and self.configer.get('iters') == 0:
            self.__val()

        self.module_utilizer.set_status(self.det_net, status='train')
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.configer.plus_one('epoch')
        self.scheduler.step(self.configer.get('epoch'))

        # data_tuple: (inputs, heatmap, maskmap, vecmap)
        for i, (inputs, batch_gt_bboxes, batch_gt_labels) in enumerate(self.train_loader):
            bboxes, labels = self.det_data_utilizer.ssd_batch_encode(batch_gt_bboxes,
                                                                     batch_gt_labels,
                                                                     self.ssd_priorbox_layer())
            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs, bboxes, labels = self.module_utilizer.to_device(inputs, bboxes, labels)

            # Forward pass.
            loc, cls = self.det_net(inputs)

            # Compute the loss of the train batch & backward.
            loss = self.det_loss(loc, bboxes, cls, labels)

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
        self.module_utilizer.set_status(self.det_net, status='val')
        start_time = time.time()
        with torch.no_grad():
            for j, (inputs, batch_gt_bboxes, batch_gt_labels) in enumerate(self.val_loader):
                # Change the data type.
                bboxes, labels = self.det_data_utilizer.ssd_batch_encode(batch_gt_bboxes,
                                                                         batch_gt_labels,
                                                                         self.ssd_priorbox_layer())
                inputs, bboxes, labels = self.module_utilizer.to_device(inputs, bboxes, labels)

                # Forward pass.
                loc, cls = self.det_net(inputs)

                # Compute the loss of the val batch.
                loss = self.det_loss(loc, bboxes, cls, labels)
                self.val_losses.update(loss.item(), inputs.size(0))
                batch_pred_bboxes = list()
                for i in range(inputs.size(0)):
                    bbox = loc.cpu().data.squeeze(0)
                    cls = F.softmax(cls.cpu().squeeze(0), dim=-1).data
                    boxes, lbls, scores = self.__decode(bbox, cls)
                    pred_bboxes = self.__get_object_list(boxes, lbls, scores)
                    batch_pred_bboxes.append(pred_bboxes)

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
            self.module_utilizer.set_status(self.det_net, status='train')

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
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0]
            keep.append(i)
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
            keep = self.__nms(boxes[ids], max_conf[ids])

            pred_bboxes = boxes[ids][keep].cpu().numpy()
            pred_bboxes = np.clip(pred_bboxes, 0, 1)
            pred_labels = labels[ids][keep].cpu().numpy()
            pred_confs = max_conf[ids][keep].cpu().numpy()

            return pred_bboxes, pred_labels, pred_confs

        else:
            Log.info('None object detected!')
            pred_bboxes = list()
            pred_labels = list()
            pred_confs = list()
            return pred_bboxes, pred_labels, pred_confs

    def __get_object_list(self, box_list, label_list, conf):
        object_list = list()
        for bbox, label, cf in zip(box_list, label_list, conf):
            if cf < self.configer.get('vis', 'conf_threshold'):
                continue

            xmin = bbox[0]
            xmax = bbox[2]
            ymin = bbox[1]
            ymax = bbox[3]
            object_list.append([xmin, ymin, xmax, ymax, label - 1, float('%.2f' % cf)])

        return object_list

    def train(self):
        cudnn.benchmark = True
        while self.configer.get('epoch') < self.configer.get('solver', 'max_epoch'):
            self.__train()
            if self.configer.get('epoch') == self.configer.get('solver', 'max_epoch'):
                break


if __name__ == "__main__":
    # Test class for pose estimator.
    pass