#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Multibox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn


class AADetectionLayer(nn.Module):

    def __init__(self, configer):
        super(AADetectionLayer, self).__init__()

        self.num_classes = configer.get('data', 'num_classes')
        self.num_anchors = configer.get('gt', 'num_anchor_list')
        self.num_features = configer.get('network', 'num_feature_list')

        self.anchor_layers = nn.ModuleList()

        for i in range(len(self.num_anchors)):
            self.anchor_layers.append(
                nn.Conv2d(self.num_features[i], self.num_anchors[i] * 2, kernel_size=3, padding=1)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat_list):
        anchors = []

        for i, x in enumerate(feat_list):
            anchor = self.anchor_layers[i](x)
            N = anchor.size(0)
            anchor = anchor.permute(0, 2, 3, 1).contiguous()
            anchor = anchor.view(N, -1, 2)
            anchors.append(anchor)

        anchor_preds = torch.cat(anchors, 1)

        return anchor_preds
