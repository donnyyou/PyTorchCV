#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Faster R-CNN.


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
from methods.det.faster_rcnn_test import FastRCNNTest
from models.det_model_manager import DetModelManager
from utils.layers.det.fr_priorbox_layer import FRPriorBoxLayer
from utils.layers.det.fr_roi_generator import FRRoiGenerator
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from val.scripts.det.det_running_score import DetRunningScore
from vis.visualizer.det_visualizer import DetVisualizer


class FasterRCNN(object):
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
        self.fr_priorbox_layer = FRPriorBoxLayer(configer)
        self.fr_roi_generator = FRRoiGenerator(configer)
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

        self.fr_loss = self.det_loss_manager.get_det_loss('fr_loss')

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
            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs = self.module_utilizer.to_device(inputs)

            # Forward pass.
            feat = self.det_net.extractor(inputs)
            rpn_locs, rpn_scores = self.det_net.rpn(feat)
            train_indices_and_rois = self.fr_roi_generator(rpn_locs, rpn_scores,
                                                           self.configer.get('rpn', 'n_train_pre_nms'),
                                                           self.configer.get('rpn', 'n_train_post_nms'))

            gt_rpn_locs, gt_rpn_labels = self.det_data_utilizer.rpn_batch_encode(
                batch_gt_bboxes, self.fr_priorbox_layer())
            gt_rpn_locs, gt_rpn_labels = self.module_utilizer.to_device(gt_rpn_locs, gt_rpn_labels)

            sample_rois, gt_roi_bboxes, gt_roi_labels = self.det_data_utilizer.roi_batch_encode(
                batch_gt_bboxes, batch_gt_labels, indices_and_rois=train_indices_and_rois.cpu())
            sample_rois, gt_roi_bboxes, gt_roi_labels = self.module_utilizer.to_device(sample_rois,
                                                                                       gt_roi_bboxes, gt_roi_labels)

            sample_roi_locs, sample_roi_scores = self.det_net.roi_head(feat, sample_rois)
            sample_roi_locs = sample_roi_locs.contiguous().view(-1, self.configer.get('data', 'num_classes'), 4)
            sample_roi_locs = sample_roi_locs[
                torch.arange(0, sample_roi_locs.size()[0]).long().to(sample_roi_locs.device),
                gt_roi_labels.long().to(sample_roi_locs.device)].contiguous().view(-1, 4)

            # Compute the loss of the train batch & backward.
            loss = self.fr_loss([rpn_locs, rpn_scores, sample_roi_locs, sample_roi_scores],
                                [gt_rpn_locs, gt_rpn_labels, gt_roi_bboxes, gt_roi_labels])

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

                # Forward pass.
                feat = self.det_net.extractor(inputs)
                rpn_locs, rpn_scores = self.det_net.rpn(feat)
                train_indices_and_rois = self.fr_roi_generator(rpn_locs, rpn_scores,
                                                               self.configer.get('rpn', 'n_train_pre_nms'),
                                                               self.configer.get('rpn', 'n_train_post_nms'))

                test_indices_and_rois = self.fr_roi_generator(rpn_locs, rpn_scores,
                                                              self.configer.get('rpn', 'n_test_pre_nms'),
                                                              self.configer.get('rpn', 'n_test_post_nms'))

                gt_rpn_locs, gt_rpn_labels = self.det_data_utilizer.rpn_batch_encode(
                    batch_gt_bboxes, self.fr_priorbox_layer())
                gt_rpn_locs, gt_rpn_labels = self.module_utilizer.to_device(gt_rpn_locs, gt_rpn_labels)

                sample_rois, gt_roi_bboxes, gt_roi_labels = self.det_data_utilizer.roi_batch_encode(
                    batch_gt_bboxes, batch_gt_labels, indices_and_rois=train_indices_and_rois.cpu())
                sample_rois, gt_roi_bboxes, gt_roi_labels = self.module_utilizer.to_device(sample_rois,
                                                                                           gt_roi_bboxes,
                                                                                           gt_roi_labels)

                sample_roi_locs, sample_roi_scores = self.det_net.roi_head(feat, sample_rois)
                sample_roi_locs = sample_roi_locs.contiguous().view(-1, self.configer.get('data', 'num_classes'), 4)
                sample_roi_locs = sample_roi_locs[
                    torch.arange(0, sample_roi_locs.size()[0]).long().to(sample_roi_locs.device),
                    gt_roi_labels.long().to(sample_roi_locs.device)].contiguous().view(-1, 4)

                test_roi_locs, test_roi_scores = self.det_net.roi_head(feat, test_indices_and_rois)

                # Compute the loss of the train batch & backward.

                loss = self.fr_loss([rpn_locs, rpn_scores, sample_roi_locs, sample_roi_scores],
                                    [gt_rpn_locs, gt_rpn_labels,  gt_roi_bboxes, gt_roi_labels])

                self.val_losses.update(loss.item(), inputs.size(0))
                batch_detections = FastRCNNTest.decode(test_roi_locs,
                                                       test_roi_scores,
                                                       test_indices_and_rois,
                                                       self.configer,
                                                       inputs.size(0))
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
            Log.info('Val mAP: {}\n'.format(self.det_running_score.get_mAP()))
            self.det_running_score.reset()
            self.batch_time.reset()
            self.val_losses.reset()
            self.module_utilizer.set_status(self.det_net, status='train')

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
                    cls_pred = int(cls_pred.cpu().item()) - 1
                    object_list.append([xmin, ymin, xmax, ymax, cls_pred, float('%.2f' % cf)])

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