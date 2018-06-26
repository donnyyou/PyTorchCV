#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Multibox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn


class MultiBoxLayer(nn.Module):

    def __init__(self, configer):
        super(MultiBoxLayer, self).__init__()

        self.num_classes = configer.get('data', 'num_classes')
        self.num_anchors = configer.get('details', 'num_anchor_list')
        self.num_features = configer.get('details', 'num_feature_list')

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.num_features)):
            self.loc_layers.append(
                nn.Conv2d(self.num_features[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)
            )
            self.conf_layers.append(
                nn.Conv2d(self.num_features[i], self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)
            )

    def forward(self, features):
        y_locs = []
        y_confs = []

        for i, x in enumerate(features):
            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)

        return loc_preds, conf_preds


class ShareMultiBoxLayer(nn.Module):

    def __init__(self, configer):
        super(ShareMultiBoxLayer, self).__init__()

        self.num_classes = configer.get('data', 'num_classes')
        self.num_anchors = configer.get('details', 'num_anchor_list')
        self.num_features = configer.get('details', 'num_feature_list')

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        for i in range(2):
            self.loc_layers.append(
                nn.Conv2d(self.num_features[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)
            )
            self.conf_layers.append(
                nn.Conv2d(self.num_features[i], self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)
            )

    def forward(self, features):
        y_locs = []
        y_confs = []

        for i, x in enumerate(features):
            if i == 0:
                y_loc = self.loc_layers[0](x)
                y_conf = self.conf_layers[0](x)
            else:
                y_loc = self.loc_layers[1](x)
                y_conf = self.conf_layers[1](x)

            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(y_loc.size(0), -1, 4)
            y_locs.append(y_loc)

            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(y_conf.size(0), -1, self.num_classes)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)

        return loc_preds, conf_preds


class UnifiedMultiBoxLayer(nn.Module):

    def __init__(self, configer):
        super(UnifiedMultiBoxLayer, self).__init__()

        self.num_classes = configer.get('data', 'num_classes')
        self.num_anchors = configer.get('details', 'num_anchor_list')
        self.num_features = configer.get('details', 'num_feature_list')

        self.loc_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=self.num_anchors[0] * 4, kernel_size=1)
        )
        self.conf_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=self.num_anchors[0] * self.num_classes, kernel_size=1)
        )

    def forward(self, features):
        y_locs = []
        y_confs = []

        for x in enumerate(features):
            y_loc = self.loc_layer(x[1])
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = self.conf_layer(x[1])
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)

        return loc_preds, conf_preds
