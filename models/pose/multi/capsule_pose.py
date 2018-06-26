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


class CapsulePose(nn.Module):
    def __init__(self, configer):
        super(CapsulePose, self).__init__()

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
        self.last_conv = nn.Conv2d(self.in_channels, self.configer.get('network', 'partmap_out'),
                                   kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        partmap_out = self.last_conv(x)

        return partmap_out


if __name__ == "__main__":
    pass