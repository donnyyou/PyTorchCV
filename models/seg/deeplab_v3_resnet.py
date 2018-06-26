#!/usr/bin/env python
# -*- coding:utf-8 -*-
# deeplabv3 res101 (synchronized BN version)
# Author: Xiangtai(lxtpku@pku.edu.cn)

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplab_resnet_synbn import ModelBuilder, _ConvBatchNormReluBlock, _ResidualBlockMulGrid
from extensions.layers.nn import SynchronizedBatchNorm2d


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module with image pool (Deeplabv3)"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module(
            'c0',
            _ConvBatchNormReluBlock(in_channels, out_channels, 1, 1, 0, 1),
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                'c{}'.format(i + 1),
                _ConvBatchNormReluBlock(
                    in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
                 nn.AdaptiveAvgPool2d(1),
                _ConvBatchNormReluBlock(
                    in_channels, out_channels, 1, 1, 0, 1)
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.upsample(h, size=x.shape[2:], mode='bilinear')]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h


class DeepLabV3(nn.Module):
    def __init__(self, num_classes, multi_grid):
        super(DeepLabV3, self).__init__()
        self.resnet_features = ModelBuilder().build_encoder("resnet101")
        self.features = nn.Sequential(
            self.resnet_features.conv1, self.resnet_features.bn1, self.resnet_features.relu1,
            self.resnet_features.conv2, self.resnet_features.bn2, self.resnet_features.relu2,
            self.resnet_features.conv3, self.resnet_features.bn3, self.resnet_features.relu3, self.resnet_features.maxpool,
            self.resnet_features.layer1, self.resnet_features.layer2, self.resnet_features.layer3
        )
        self.MG_features = _ResidualBlockMulGrid(
            layers=3, inplanes=1024, midplanes=512, outplanes=2048, stride=1, dilation=2, mulgrid=multi_grid)
        pyramids = [6, 12, 18]
        self.aspp = _ASPPModule(2048, 256, pyramids)

        self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1),  # 256 * 5 = 1280
                                 SynchronizedBatchNorm2d(256))
        self.fc2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.MG_features(x)
        x = self.aspp(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.upsample(x, scale_factor=(16, 16), mode="bilinear")
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()


if __name__ == '__main__':
    model = DeepLabV3(20, multi_grid=[1, 2, 1])
    model.freeze_bn()
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True)
    print(type(model.resnet_features))
    print (model(image).size())
