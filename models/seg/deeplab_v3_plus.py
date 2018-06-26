#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai(lxtpku@pku.edu.cn)
# DeepLabv3_plus


import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplab_resnet_synbn import _ResidualBlockMulGrid, ModelBuilder
from deeplab_v3_resnet import _ASPPModule

from utils.tools.logger import Logger as Log

class DeepLabV3Plus(nn.Sequential):
    """DeepLab v3+ """
    def __init__(self, configer, pyramids=[6, 12, 18], multi_grid=[1, 2, 4]):
        super(DeepLabV3Plus, self).__init__()
        self.num_classes = configer.get('network', 'out_channels')
        self.resnet_features = ModelBuilder().build_encoder("resnet101_dilated16")
        self.low_features = nn.Sequential(
            self.resnet_features.conv1, self.resnet_features.bn1, self.resnet_features.relu1,
            self.resnet_features.conv2, self.resnet_features.bn2, self.resnet_features.relu2,
            self.resnet_features.conv3, self.resnet_features.bn3, self.resnet_features.relu3,
            self.resnet_features.maxpool,
            self.resnet_features.layer1,
        )
        self.high_features = nn.Sequential(self.resnet_features.layer2, self.resnet_features.layer3)
        self.MG_features = _ResidualBlockMulGrid(layers=3, inplanes=1024, midplanes=512, outplanes=2048, stride=1,
                                                 dilation=2, mulgrid=multi_grid)
        self.aspp = _ASPPModule(2048, 256, pyramids)
        self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1),  # 256 * 5 = 1280
                                 nn.BatchNorm2d(256))

        self.reduce_conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1),
                                          nn.BatchNorm2d(48))
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        Log.info(x.size())
        low = self.low_features(x)
        x = self.high_features(low)
        x = self.MG_features(x)
        x = self.aspp(x)
        x = self.fc1(x)
        x = F.upsample(x, scale_factor=(4, 4), mode='bilinear')
        low = self.reduce_conv2(low)

        x = torch.cat((x, low), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=(4, 4), mode='bilinear')

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

if __name__ == '__main__':
    model = DeepLabV3Plus(pyramids=[6, 12, 18]).cuda()
    model.freeze_bn()
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True).cuda()
    print (model(image).size())