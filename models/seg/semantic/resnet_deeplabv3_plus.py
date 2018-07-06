#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Pytorch implementation of DeepLabv3_plus based on resnet101(torch version) ,Synchronized Batch Normalization


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone_selector import BackboneSelector
from extensions.layers.encoding.syncbn import BatchNorm2d


class _ConvBatchNormReluBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.relu = relu
        self.conv =  nn.Conv2d(in_channels=inplanes,out_channels=outplanes,
                            kernel_size=kernel_size, stride=stride, padding = padding,
                            dilation = dilation, bias=False)
        self.bn = BatchNorm2d(num_features=outplanes)
        self.relu_f = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.relu:
            x = self.relu_f(x)
        return x


class _Bottleneck(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBatchNormReluBlock(inplanes, midplanes, 1, stride, 0, 1)
        self.conv3x3 = _ConvBatchNormReluBlock(midplanes, midplanes, 3, 1, dilation, dilation)
        self.increase = _ConvBatchNormReluBlock(midplanes, outplanes, 1, 1, 0, 1, relu=False)
        self.downsample = downsample
        if self.downsample:
            self.proj = _ConvBatchNormReluBlock(inplanes, outplanes, 1, stride, 0, 1, relu=False)

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class _ResidualBlockMulGrid(nn.Module):
    def __init__(self,inplanes, midplanes, outplanes, stride, dilation, mulgrid=[1,2,1]):
        super(_ResidualBlockMulGrid, self).__init__()
        self.block1 = _Bottleneck(inplanes, midplanes, outplanes, stride, dilation * mulgrid[0], True)
        self.block2 = _Bottleneck(outplanes, midplanes, outplanes, 1, dilation * mulgrid[1], False)
        self.block3 = _Bottleneck(outplanes, midplanes, outplanes, 1, dilation * mulgrid[2], False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# ASPP module
class Atrous_module(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, rate):
        super(Atrous_module, self).__init__()
        self.conv = nn.Conv2d(inplanes,midplanes,kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(midplanes)
        self.atrous_convolution = nn.Conv2d(midplanes, outplanes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.atrous_convolution(x)

        return x


class ResNetDeepLabV3Plus(nn.Sequential):
    """DeepLab v3+ """
    def __init__(self, configer):
        super(ResNetDeepLabV3Plus, self).__init__()
        self.configer = configer
        pyramids=[1, 6, 12, 18]
        multi_grid=[1, 2, 4]
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        num_features = self.backbone.get_num_features()

        self.low_features = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
        )
        self.high_features = nn.Sequential(self.backbone.layer2, self.backbone.layer3)
        self.MG_features = _ResidualBlockMulGrid(inplanes=1024, midplanes=512, outplanes=2048, stride=2,
                                                 dilation=2, mulgrid=multi_grid)
        self.aspp1 = Atrous_module(2048, 256, 256, rate=pyramids[0])
        self.aspp2 = Atrous_module(2048, 256, 256, rate=pyramids[1])
        self.aspp3 = Atrous_module(2048, 256, 256, rate=pyramids[2])
        self.aspp4 = Atrous_module(2048, 256, 256, rate=pyramids[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(2048, 256, kernel_size=1))

        self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1),  # 256 * 5 = 1280
                                 BatchNorm2d(256), nn.ReLU(inplace=True))

        self.reduce_conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1),
                                          BatchNorm2d(48),
                                          nn.ReLU(inplace=True)
                                          )
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(0.1),
                                       nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        low = self.low_features(x)
        x = self.high_features(low)
        # print("high", x.size())
        x = self.MG_features(x)
        # print("MG",x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x6 = F.upsample(x5, size=x.size()[2:], mode="bilinear")
        x = torch.cat((x1, x2, x3, x4, x6), dim=1)
        # print(x.size())
        x = self.fc1(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')
        low = self.reduce_conv2(low)
        # print(low.size(),x.size())
        x = torch.cat((x, low), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')

        out = []
        out.append(x)
        return tuple(out)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, BatchNorm2d):
                m.eval()

if __name__ == '__main__':
    i = torch.Tensor(1,3,512,512).cuda()
    model = ResNetDeepLabV3Plus(num_classes=19,pretrained_dir="/home/xiangtai/pretrained/resnet101-5d3b4d8f.pth").cuda()
    o = model(i)
    print(o)

"""
Here are some suggestions:
1, Set atrous_rates [6, 12, 18] to [12, 24, 36] if setting output_stride=8.(big feature map)
2, train over 90K on both coarse and fine data on cityscape
"""