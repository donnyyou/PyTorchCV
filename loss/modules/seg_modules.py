#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, configer=None):
        super(CrossEntropyLoss, self).__init__()
        self.configer = configer
        weight = None
        if not self.configer.is_empty('cross_entropy_loss', 'weight'):
            weight = self.configer.get('cross_entropy_loss', 'weight')

        size_average = self.configer.get('cross_entropy_loss', 'size_average')
        reduce = self.configer.get("cross_entropy_loss", "reduce")

        self.nll_loss = nn.NLLLoss(weight=weight, size_average=size_average, reduce=reduce)

    def forward(self, inputs, targets, maskmap=None):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class FocalLoss(nn.Module):
    def __init__(self, configer):
        super(FocalLoss, self).__init__()
        self.configer = configer
        self.y = self.configer.get('focal_loss', 'y')

    def forward(self, output, target):
        P = F.softmax(output)
        f_out = F.log_softmax(output)
        Pt = P.gather(1, torch.unsqueeze(target, 1))
        focus_p = torch.pow(1 - Pt, self.y)
        alpha = 0.25
        nll_feature = -f_out.gather(1, torch.unsqueeze(target, 1))
        weight_nll = alpha * focus_p * nll_feature
        loss = weight_nll.mean()
        return loss


class Ege_loss(nn.Module):
    def __init__(self):
        super(Ege_loss, self).__init__()

    def forward(self, output, target):
        target = self.transform_target(target)
        return F.mse_loss(output,target)

    def make_kernel(self):
        k1 = torch.FloatTensor([1, 0, 0, 0, -1, 0, 0, 0, 0]).view(3, 3)
        k2 = torch.FloatTensor([0, 1, 0, 0, -1, 0, 0, 0, 0]).view(3, 3)
        k3 = torch.FloatTensor([0, 0, 1, 0, -1, 0, 0, 0, 0]).view(3, 3)
        k4 = torch.FloatTensor([0, 0, 0, 1, -1, 0, 0, 0, 0]).view(3, 3)
        k5 = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0]).view(3, 3)
        k6 = torch.FloatTensor([0, 0, 0, 0, -1, 1, 0, 0, 0]).view(3, 3)
        k7 = torch.FloatTensor([0, 0, 0, 0, -1, 0, 1, 0, 0]).view(3, 3)
        k8 = torch.FloatTensor([0, 0, 0, 0, -1, 0, 0, 1, 0]).view(3, 3)
        k9 = torch.FloatTensor([0, 0, 0, 0, -1, 0, 0, 0, 1]).view(3, 3)
        kernel = torch.cat((k1, k2, k3, k4, k5, k6, k7, k8, k9), dim=0).view(-1, 3, 3).unsqueeze(1)
        return kernel

    def make_vector(self):
        v1 = torch.FloatTensor([-1, 1])
        v2 = torch.FloatTensor([0, 1])
        v3 = torch.FloatTensor([1, 1])
        v4 = torch.FloatTensor([-1, 0])
        v5 = torch.FloatTensor([0, 0])
        v6 = torch.FloatTensor([1, 0])
        v7 = torch.FloatTensor([-1, -1])
        v8 = torch.FloatTensor([0, -1])
        v9 = torch.FloatTensor([1, -1])
        vec = torch.cat((v1, v2, v3, v4, v5, v6, v7, v8, v9), dim=0).view(-1, 2).unsqueeze(2).unsqueeze(3)
        return vec

    def transform_target(self, target, dilation=1):
        b, h, w = target.size()
        kernel = self.make_kernel()
        vec = self.make_vector()
        target = target.unsqueeze(1)
        out = F.conv2d(target, weight=kernel, padding=dilation, dilation=dilation)
        mask1 = out != 0
        mask2 = out == 0
        out[mask1] = 0
        out[mask2] = 1
        out = out.repeat(1, 2, 1, 1).view(b, 9, 2, h, w) * vec
        return out.sum(dim=1)