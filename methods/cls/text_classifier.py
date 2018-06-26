#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Image Classifier.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, lr_scheduler

from vis.visdom.visdom_helper import VisdomHelper
from datasets.cls.tc_data_loader import TCDataLoader
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.cross_entropy_loss(inputs, targets)


class ClsModel(nn.Module):
    def __init__(self):
        super(ClsModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=(3, 300))
        self.relu1 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(100, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TextClassifier(object):
    """
      The class for the training phase of Image classification.
    """
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.log_stream = open('./log.txt', 'w')
        self.cls_net = None
        self.train_loader = None
        self.val_loader = None
        self.batch = 0
        self.max_batch = 500000


    def init_model(self):
        self.cls_net = ClsModel()

        self.ce_loss = CrossEntropyLoss()
        self.train_batch_loader = TCDataLoader('/home/donny/DataSet/Text/train', batch_size=6)
        self.val_batch_loader = TCDataLoader('/home/donny/DataSet/Text/val', batch_size=1)

        # self.optimizer = SGD(self.cls_net.parameters(), lr=0.01, momentum=0.9,
        #                     weight_decay=0.0001)
        self.optimizer = Adam(self.cls_net.parameters(), lr=0.00004, betas = [0.9, 0.999],
                              eps = 1e-08, weight_decay=0.0001)

        self.scheduler = lr_scheduler.StepLR(self.optimizer, 100000, gamma=0.333)

    def _get_parameters(self):

        return self.cls_net.parameters()

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.cls_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.

        while self.batch < self.max_batch:
            self.data_time.update(time.time() - start_time)
            # Change the data type.
            inputs, labels = self.train_batch_loader.next_batch()
            # Forward pass.
            outputs = self.cls_net(inputs)
            # Compute the loss of the train batch & backward.

            loss = self.ce_loss(outputs, labels)

            self.train_losses.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.batch += 1
            self.scheduler.step(self.batch)
            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

            # Print the log info & reset the states.
            if self.batch % 50 == 0:
                Log.info('Train Batch: {0}\t'
                         'Time {batch_time.sum:.3f}s / 50 iters, ({batch_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / 50 iters, ({data_time.avg:3f})\n'
                         'Learning rate = {1}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    self.batch,
                    self.scheduler.get_lr(), batch_time=self.batch_time,
                    data_time=self.data_time, loss=self.train_losses))

                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

                self.__val()

        self.log_stream.close()

    def __val(self):
        """
          Validation function during the train phase.
        """
        self.cls_net.eval()
        index = 0
        acc_count = 0

        while index < self.val_batch_loader.num_data():
            # Change the data type.
            inputs, labels = self.val_batch_loader.next_batch()
            # Forward pass.
            outputs = self.cls_net(inputs)

            pred = outputs.data.max(1)[1]
            pred = pred.squeeze()
            if pred.item() == labels.item():
                acc_count += 1

            index += 1

        Log.info('Top1 ACC = {}\n'.format(acc_count / index))

        self.log_stream.write('{}, {}\n'.format(self.batch, acc_count / index))
        self.log_stream.flush()
        self.cls_net.train()


if __name__ == "__main__":
    # Test class for pose estimator.
    text_classifier = TextClassifier()
    text_classifier.init_model()
    text_classifier.train()