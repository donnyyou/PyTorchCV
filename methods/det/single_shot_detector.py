#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.backends.cudnn as cudnn

from datasets.det_data_loader import DetDataLoader
from datasets.det.det_data_utilizer import DetDataUtilizer
from loss.det_loss_manager import DetLossManager
from methods.tools.module_utilizer import ModuleUtilizer
from methods.tools.optim_scheduler import OptimScheduler
from methods.det.single_shot_detector_test import SingleShotDetectorTest
from models.det_model_manager import DetModelManager
from utils.layers.det.ssd_priorbox_layer import SSDPriorBoxLayer
from utils.layers.det.ssd_anchor_target_layer import SSDAnchorTargetLayer
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
        self.ssd_anchor_target_layer = SSDAnchorTargetLayer(configer)
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
            # Change the data type.
            inputs = self.module_utilizer.to_device(inputs)

            self.data_time.update(time.time() - start_time)
            # Forward pass.
            feat_list, loc, cls = self.det_net(inputs)

            bboxes, labels = self.ssd_anchor_target_layer(feat_list, batch_gt_bboxes,
                                                          batch_gt_labels, [inputs.size(3), inputs.size(2)])

            bboxes, labels = self.module_utilizer.to_device(bboxes, labels)
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
                inputs = self.module_utilizer.to_device(inputs)
                input_size = [inputs.size(3), inputs.size(2)]
                # Forward pass.
                feat_list, loc, cls = self.det_net(inputs)
                bboxes, labels = self.ssd_anchor_target_layer(feat_list, batch_gt_bboxes,
                                                              batch_gt_labels, input_size)

                bboxes, labels = self.module_utilizer.to_device(bboxes, labels)
                # Compute the loss of the val batch.
                loss = self.det_loss(loc, bboxes, cls, labels)
                self.val_losses.update(loss.item(), inputs.size(0))

                batch_detections = SingleShotDetectorTest.decode(loc, cls,
                                                                 self.ssd_priorbox_layer(feat_list, input_size),
                                                                 self.configer)
                batch_pred_bboxes = self.__get_object_list(batch_detections)
                # batch_pred_bboxes = self._get_gt_object_list(batch_gt_bboxes, batch_gt_labels)
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

    def _get_gt_object_list(self, batch_gt_bboxes, batch_gt_labels):
        batch_pred_bboxes = list()
        for i in range(len(batch_gt_bboxes)):
            object_list = list()
            if batch_gt_bboxes[i].numel() > 0:
                for j in range(batch_gt_bboxes[i].size(0)):
                    object_list.append([batch_gt_bboxes[i][j][0].item(), batch_gt_bboxes[i][j][1].item(),
                                        batch_gt_bboxes[i][j][2].item(), batch_gt_bboxes[i][j][3].item(),
                                        batch_gt_labels[i][j].item(), 1.0])

            batch_pred_bboxes.append(object_list)
        return batch_pred_bboxes

    def __get_object_list(self, batch_detections):
        batch_pred_bboxes = list()
        for idx, detections in enumerate(batch_detections):
            object_list = list()
            if detections is not None:
                for x1, y1, x2, y2, conf, cls_pred in detections:
                    xmin = x1.cpu().item()
                    ymin = y1.cpu().item()
                    xmax = x2.cpu().item()
                    ymax = y2.cpu().item()
                    cf = conf.cpu().item()
                    cls_pred = cls_pred.cpu().item() - 1
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