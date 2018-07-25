#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
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
            weight = torch.FloatTensor(weight).cuda()

        size_average = True
        if not self.configer.is_empty('cross_entropy_loss', 'size_average'):
            size_average = self.configer.get('cross_entropy_loss', 'size_average')

        reduce = True
        if not self.configer.is_empty('cross_entropy_loss', 'reduce'):
            reduce = self.configer.get("cross_entropy_loss", "reduce")

        ignore_index = -100
        if not self.configer.is_empty('cross_entropy_loss', 'ignore_index'):
            ignore_index = self.configer.get('cross_entropy_loss', 'ignore_index')

        self.nll_loss = nn.NLLLoss(weight=weight,
                                   size_average=size_average,
                                   ignore_index=ignore_index,
                                   reduce=reduce)

    def forward(self, inputs, targets, weights=None):
        loss = 0.0
        if isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if isinstance(targets, list):
                    loss += weights[i] * self.nll_loss(F.log_softmax(inputs[i], dim=1), targets[i])
                else:
                    loss += weights[i] * self.nll_loss(F.log_softmax(inputs[i], dim=1), targets)

        else:
            loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, configer):
        super(FocalLoss, self).__init__()
        self.configer = configer

    def forward(self, output, target):
        self.y = self.configer.get('focal_loss', 'y')
        P = F.softmax(output)
        f_out = F.log_softmax(output)
        Pt = P.gather(1, torch.unsqueeze(target, 1))
        focus_p = torch.pow(1 - Pt, self.y)
        alpha = 0.25
        nll_feature = -f_out.gather(1, torch.unsqueeze(target, 1))
        weight_nll = alpha * focus_p * nll_feature
        loss = weight_nll.mean()
        return loss


class SegEncodeLoss(nn.Module):
    def __init__(self, configer):
        super(SegEncodeLoss, self).__init__()
        self.configer = configer
        weight = None
        if not self.configer.is_empty('seg_encode_loss', 'weight'):
            weight = self.configer.get('seg_encode_loss', 'weight')
            weight = torch.FloatTensor(weight).cuda()

        size_average = True
        if not self.configer.is_empty('seg_encode_loss', 'size_average'):
            size_average = self.configer.get('seg_encode_loss', 'size_average')

        reduce = True
        if not self.configer.is_empty('seg_encode_loss', 'reduce'):
            reduce = self.configer.get("seg_encode_loss", "reduce")

        self.bce_loss = nn.BCELoss(weight, size_average, reduce=reduce)

    def forward(self, preds, targets, grid_scale=None):
        if len(targets.size()) == 2:
            return self.bce_loss(F.sigmoid(preds), targets)

        se_target = self._get_batch_label_vector(targets,
                                                 self.configer.get('data', 'num_classes'),
                                                 grid_scale).type_as(preds)
        return self.bceloss(F.sigmoid(preds), se_target)

    @staticmethod
    def _get_batch_label_vector(target_, num_classes, grid_scale=None):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        b, h, w = target_.size()
        pad_h = 0 if (h % 6 == 0) else 6 - (h % 6)
        pad_w = 0 if (w % 6 == 0) else 6 - (w % 6)
        target = target_.clone()
        target = F.pad(target, (0, pad_w, 0, pad_h), "constant", num_classes)

        if grid_scale is not None:
            target = target.contiguous().view(b, h // grid_scale, grid_scale, w // grid_scale, grid_scale)
            target = target.permute(0, 2, 4, 1, 3).contiguous().view(b * grid_scale * grid_scale,
                                                                        h // grid_scale, w // grid_scale)

        batch = target.size(0)
        tvect = torch.zeros(batch, num_classes)
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=num_classes, min=0, max=num_classes - 1)
            vect = hist>0
            tvect[i] = vect

        return tvect


class FCNSegLoss(nn.CrossEntropyLoss):
    def __init__(self, configer):
        super(FCNSegLoss, self).__init__()
        self.configer = configer
        self.ce_loss = CrossEntropyLoss(self.configer)
        self.se_loss = SegEncodeLoss(self.configer)
        self.focal_loss = FocalLoss(self.configer)

    def forward(self, *outputs):
        if self.configer.get('network', 'model_name') == 'pyramid_encnet':
            seg_out, se_out_list, aux_out, targets = outputs
            seg_loss = self.ce_loss(seg_out, targets)
            aux_targets = self._scale_target(targets, (aux_out.size(2), aux_out.size(1)))
            aux_loss = self.ce_loss(aux_out, aux_targets)
            loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
            loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
            se_weight = self.configer.get('network', 'loss_weights')['se_loss']
            for i, scale in enumerate(self.configer.get('network', 'pyramid')):
                loss = loss + se_weight * self.se_loss(se_out_list[i], aux_targets, scale)

            return loss

        elif self.configer.get('network', 'model_name') == 'syncbn_pspnet':
            seg_out, aux_out, targets = outputs
            seg_loss = self.ce_loss(seg_out, targets)
            aux_targets = self._scale_target(targets, (aux_out.size(3), aux_out.size(2)))
            aux_loss = self.ce_loss(aux_out, aux_targets)
            loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
            loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
            return loss

        return self.ce_loss(outputs, targets)

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().permute(1, 2, 0).cpu().numpy()
        targets = cv2.resize(targets, (0, 0),
                             fx=scaled_size[1] / targets.shape[0],
                             fy=scaled_size[0] / targets.shape[1],
                             interpolation=cv2.INTER_NEAREST)
        targets = torch.from_numpy(targets.transpose(2, 0, 1)).long().to(targets_.device)
        return targets


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
