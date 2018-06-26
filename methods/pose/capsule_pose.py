#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Pose Estimator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.backends.cudnn as cudnn

from datasets.pose_data_loader import PoseDataLoader
from loss.pose_loss_manager import PoseLossManager
from methods.tools.module_utilizer import ModuleUtilizer
from methods.tools.optim_scheduler import OptimScheduler
from models.pose_model_manager import PoseModelManager
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from vis.visualizer.pose_visualizer import PoseVisualizer


class CapsulePose(object):
    """
      The class for Pose Estimation. Include train, val, test & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.train_loss_heatmap = AverageMeter()
        self.train_loss_associate = AverageMeter()
        self.val_losses = AverageMeter()
        self.val_loss_heatmap = AverageMeter()
        self.val_loss_associate = AverageMeter()
        self.pose_visualizer = PoseVisualizer(configer)
        self.pose_loss_manager = PoseLossManager(configer)
        self.pose_model_manager = PoseModelManager(configer)
        self.pose_data_loader = PoseDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.optim_scheduler = OptimScheduler(configer)

        self.pose_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

    def init_model(self):
        self.pose_net = self.pose_model_manager.multi_pose_detector()
        self.pose_net = self.module_utilizer.load_net(self.pose_net)

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(self._get_parameters())

        self.train_loader = self.pose_data_loader.get_trainloader()
        self.val_loader = self.pose_data_loader.get_valloader()

        self.weights = self.configer.get('network', 'weights')
        self.mse_loss = self.pose_loss_manager.get_pose_loss('mse_loss')

    def _get_parameters(self):

        return self.pose_net.parameters()

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        if self.configer.get('network', 'resume') is not None and self.configer.get('iters') == 0:
            self.__val()

        self.pose_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.configer.plus_one('epoch')
        self.scheduler.step(self.configer.get('epoch'))

        # data_tuple: (inputs, heatmap, maskmap, vecmap)
        for i, (inputs, partmap, maskmap, vecmap) in enumerate(self.train_loader):
            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs, partmap, maskmap, vecmap = self.module_utilizer.to_device(inputs, partmap, maskmap, vecmap)

            # Forward pass.
            paf_out, partmap_out = self.pose_net(inputs)

            # Compute the loss of the train batch & backward.
            loss_heatmap = self.mse_loss(partmap_out, partmap, mask=maskmap, weights=self.weights)
            loss_associate = self.mse_loss(paf_out, vecmap, mask=maskmap, weights=self.weights)
            loss = loss_heatmap + loss_associate

            self.train_losses.update(loss.item(), inputs.size(0))
            self.train_loss_heatmap.update(loss_heatmap.item(), inputs.size(0))
            self.train_loss_associate.update(loss_associate.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Loss Heatmap:{}, Loss Asso: {}'.format(self.train_loss_heatmap.avg,
                                                                 self.train_loss_associate.avg))
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
                self.train_loss_heatmap.reset()
                self.train_loss_associate.reset()

            # Check to val the current model.
            if self.val_loader is not None and \
               self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

    def __val(self):
        """
          Validation function during the train phase.
        """
        self.pose_net.eval()
        start_time = time.time()

        with torch.no_grad():
            for j, (inputs, heatmap, maskmap, vecmap) in enumerate(self.val_loader):
                # Change the data type.
                inputs, heatmap, maskmap, vecmap = self.module_utilizer.to_device(inputs, heatmap, maskmap, vecmap)

                # Forward pass.
                paf_out, heatmap_out = self.pose_net(inputs)
                # Compute the loss of the val batch.
                loss_heatmap = self.mse_loss(heatmap_out[-1], heatmap, maskmap)
                loss_associate = self.mse_loss(paf_out[-1], vecmap, maskmap)
                loss = loss_heatmap + loss_associate

                self.val_losses.update(loss.item(), inputs.size(0))
                self.val_loss_heatmap.update(loss_heatmap.item(), inputs.size(0))
                self.val_loss_associate.update(loss_associate.item(), inputs.size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.module_utilizer.save_net(self.pose_net, metric='iters')
            Log.info('Loss Heatmap:{}, Loss Asso: {}'.format(self.val_loss_heatmap.avg, self.val_loss_associate.avg))
            # Print the log info & reset the states.
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            self.batch_time.reset()
            self.val_losses.reset()
            self.val_loss_heatmap.reset()
            self.val_loss_associate.reset()
            self.pose_net.train()

    def train(self):
        cudnn.benchmark = True
        while self.configer.get('epoch') < self.configer.get('solver', 'max_epoch'):
            self.__train()
            if self.configer.get('epoch') == self.configer.get('solver', 'max_epoch'):
                break


if __name__ == "__main__":
    # Test class for pose estimator.
    pass