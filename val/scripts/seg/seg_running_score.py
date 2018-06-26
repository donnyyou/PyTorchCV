#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Segmentation running score.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class SegRunningScore(object):
    def __init__(self, configer):
        self.configer = configer
        self.tp_cnt = torch.zeros(self.configer.get('data', 'num_classes')).double()
        self.fp_cnt = torch.zeros(self.configer.get('data', 'num_classes')).double()
        self.fn_cnt = torch.zeros(self.configer.get('data', 'num_classes')).double()

    def get_mean_iou(self):
        # returns "iou mean", "iou per class"
        num = self.tp_cnt
        den = self.tp_cnt + self.fp_cnt + self.fn_cnt + 1e-15
        iou = num / den
        return torch.mean(iou)

    def get_class_iou(self):
        # returns "iou mean", "iou per class"
        num = self.tp_cnt
        den = self.tp_cnt + self.fp_cnt + self.fn_cnt + 1e-15
        iou = num / den
        return torch.mean(iou), iou

    def update(self, preds, targets):
        if preds.is_cuda or targets.is_cuda:
            preds = preds.cuda()
            targets = targets.cuda()

        if len(targets.size()) == 3:
            targets = targets.unsqueeze(1)

        # if size is "batch_size x 1 x H x W" scatter to onehot
        if (preds.size(1) == 1):
            x_onehot = torch.zeros(preds.size(0),
                                   self.configer.get('network', 'out_channels'),
                                   preds.size(2), preds.size(3))
            if preds.is_cuda:
                x_onehot = x_onehot.cuda()

            x_onehot.scatter_(1, preds, 1).float()

        else:
            x_onehot = preds.float()

        if (targets.size(1) == 1):
            y_onehot = torch.zeros(targets.size(0),
                                   self.configer.get('network', 'out_channels'),
                                   targets.size(2),
                                   targets.size(3))
            if targets.is_cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.scatter_(1, targets, 1).float()
        else:
            y_onehot = targets.float()

        if self.configer.get('network', 'out_channels') == self.configer.get('data', 'num_classes') + 1:
            ignores = y_onehot[:, self.configer.get('data', 'num_classes')].unsqueeze(1)
            x_onehot = x_onehot[:, :self.configer.get('data', 'num_classes')]
            y_onehot = y_onehot[:, :self.configer.get('data', 'num_classes')]
        else:
            assert self.configer.get('network', 'out_channels') == self.configer.get('data', 'num_classes')
            ignores=0

        # times prediction and gt coincide is 1
        tpmult = x_onehot * y_onehot
        tp = tpmult.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True).squeeze()

        # times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fpmult = x_onehot * (1 - y_onehot - ignores)
        fp = fpmult.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True).squeeze()

        # times prediction says its not that class and gt says it is
        fnmult = (1 - x_onehot) * (y_onehot)
        fn = fnmult.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True).squeeze()

        self.tp_cnt += tp.double().cpu()
        self.fp_cnt += fp.double().cpu()
        self.fn_cnt += fn.double().cpu()

    def reset(self):
        self.tp_cnt = torch.zeros(self.configer.get('data', 'num_classes')).double()
        self.fp_cnt = torch.zeros(self.configer.get('data', 'num_classes')).double()
        self.fn_cnt = torch.zeros(self.configer.get('data', 'num_classes')).double()