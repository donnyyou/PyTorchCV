#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.backends.cudnn as cudnn

from datasets.seg_data_loader import SegDataLoader
from loss.seg_loss_manager import SegLossManager
from methods.tools.module_utilizer import ModuleUtilizer
from methods.tools.optim_scheduler import OptimScheduler
from models.seg_model_manager import SegModelManager
from val.scripts.seg.seg_running_score import SegRunningScore
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from vis.visualizer.seg_visualizer import SegVisualizer


class FCNSegmentor(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_running_score = SegRunningScore(configer)
        self.seg_visualizer = SegVisualizer(configer)
        self.seg_loss_manager = SegLossManager(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.seg_model_manager = SegModelManager(configer)
        self.seg_data_loader = SegDataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

    def init_model(self):
        self.seg_net = self.seg_model_manager.semantic_segmentor()
        self.seg_net = self.module_utilizer.load_net(self.seg_net)

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(self._get_parameters())

        self.train_loader = self.seg_data_loader.get_trainloader()
        self.val_loader = self.seg_data_loader.get_valloader()

        self.pixel_loss = self.seg_loss_manager.get_seg_loss('cross_entropy_loss')

    def _get_parameters(self):

        return self.seg_net.parameters()

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        if self.configer.get('network', 'resume') is not None and self.configer.get('iters') == 0:
            self.__val()

        self.seg_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.configer.plus_one('epoch')
        self.scheduler.step(self.configer.get('epoch'))

        for i, (inputs, targets) in enumerate(self.train_loader):
            self.data_time.update(time.time() - start_time)
            # Change the data type.

            inputs, targets = self.module_utilizer.to_device(inputs, targets)

            # Forward pass.
            outputs = self.seg_net(inputs)

            # Compute the loss of the train batch & backward.
            loss = self.pixel_loss(outputs, targets)
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
               self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

    def __val(self):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        start_time = time.time()
        with torch.no_grad():
            for j, (inputs, targets) in enumerate(self.val_loader):
                # Change the data type.
                inputs, targets = self.module_utilizer.to_device(inputs, targets)
                # Forward pass.
                outputs = self.seg_net(inputs)
                # Compute the loss of the val batch.
                loss = self.pixel_loss(outputs, targets)

                self.val_losses.update(loss.item(), inputs.size(0))
                self.seg_running_score.update(outputs.max(1)[1].unsqueeze(1).data, targets.data)

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.configer.update_value(['performace'], self.seg_running_score.get_mean_iou())
            self.configer.update_value(['val_loss'], self.val_losses.avg)
            self.module_utilizer.save_net(self.seg_net, metric='performance')
            self.module_utilizer.save_net(self.seg_net, metric='val_loss')

            # Print the log info & reset the states.
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            Log.info('Mean IOU: {}'.format(self.seg_running_score.get_mean_iou()))
            self.batch_time.reset()
            self.val_losses.reset()
            self.seg_running_score.reset()
            self.seg_net.train()

    def train(self):
        cudnn.benchmark = True
        while self.configer.get('epoch') < self.configer.get('solver', 'max_epoch'):
            self.__train()
            if self.configer.get('epoch') == self.configer.get('solver', 'max_epoch'):
                break


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
