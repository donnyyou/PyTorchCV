#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai(lxtpku@pku.edu.cn)
# DeepLabv3_plus


import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplab_resnet_synbn import _ResidualBlockMulGrid
import seg_resnet_synbn as resnet
from deeplab_resnet_synbn import ResnetDilated
from utils.tools.logger import Logger as Log


class _ConvBatchNormReluBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.relu = relu
        self.conv =  nn.Conv2d(in_channels=inplanes,out_channels=outplanes,
                            kernel_size=kernel_size, stride=stride, padding = padding,
                            dilation = dilation, bias=False)
        self.bn = nn.BatchNorm2d(num_features=outplanes, momentum=0.999, affine=True)
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
        self.block2 = _Bottleneck(outplanes, midplanes, outplanes, stride, dilation * mulgrid[1], False)
        self.block3 =  _Bottleneck(outplanes, midplanes, outplanes, stride, dilation * mulgrid[2], False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class Atrous_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)

        return x

class DeepLabV3Plus(nn.Sequential):
    """DeepLab v3+ """
    def __init__(self, configer=None, pyramids=[1, 6, 12, 18], multi_grid=[1, 2, 4],  pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        self.num_classes = 19
        self.orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
        self.resnet_features = ResnetDilated(self.orig_resnet,dilate_scale=16)
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
        self.aspp1 = Atrous_module(2048, 256, rate=pyramids[0])
        self.aspp2 = Atrous_module(2048, 256, rate=pyramids[1])
        self.aspp3 = Atrous_module(2048, 256, rate=pyramids[2])
        self.aspp4 = Atrous_module(2048, 256, rate=pyramids[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(2048, 256, kernel_size=1))

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
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='nearest')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.fc1(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')
        low = self.reduce_conv2(low)

        x = torch.cat((x, low), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

if __name__ == '__main__':
    model = DeepLabV3Plus(pyramids=[1, 6, 12, 18]).cuda()
    model.freeze_bn()
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True).cuda()
    print (model(image).size())