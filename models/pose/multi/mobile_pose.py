#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init

from models.backbones.backbone_selector import BackboneSelector


class MobilePose(nn.Module):
    def __init__(self, configer):
        super(MobilePose, self).__init__()

        self.configer = configer
        self.backbone = BackboneSelector(configer).get_backbone()
        self.pose_model = PoseModel(configer, in_channels = self.backbone.get_num_features())

    def forward(self, x):
        x = self.backbone(x)
        out = self.pose_model(x)
        return out


class PoseModel(nn.Module):
    def __init__(self, configer,  in_channels):
        super(PoseModel, self).__init__()

        self.configer = configer
        self.in_channels = in_channels
        model_dict = self._get_model_dict()
        self.model1_1 = model_dict['block1_1']
        self.model1_2 = model_dict['block1_2']

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layers(self, layer_dict):
        layers = []

        for i in range(len(layer_dict) - 1):
            layer = layer_dict[i]
            for k in layer:
                v = layer[k]
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        layer = list(layer_dict[-1].keys())
        k = layer[0]
        v = layer_dict[-1][k]

        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
        layers += [conv2d]

        return nn.Sequential(*layers)

    def _get_model_dict(self):
        paf_out = self.configer.get('network', 'paf_out')
        heatmap_out = self.configer.get('network', 'heatmap_out')
        blocks = {}

        blocks['block1_1'] = [{'conv5_5_CPM_L1': [self.in_channels, paf_out, 1, 1, 0]}]

        blocks['block1_2'] = [{'conv5_5_CPM_L2': [self.in_channels, heatmap_out, 3, 1, 1]}]

        models = dict()

        for k in blocks:
            v = blocks[k]
            models[k] = self._make_layers(v)

        return models

    def forward(self, x):
        out1_1 = self.model1_1(x)
        out1_2 = self.model1_2(x)

        paf_out = [out1_1]
        heatmap_out = [out1_2]
        return paf_out, heatmap_out


if __name__ == "__main__":
    pass