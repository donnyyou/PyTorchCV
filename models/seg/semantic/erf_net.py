#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn as nn

from utils.tools.logger import Logger as Log


class Downsampler(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Downsampler, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes-inplanes, kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_out = self.conv1(x)
        pool_out = self.pool1(x)
        out = torch.cat([conv_out, pool_out], 1)
        out = self.relu1(out)
        return out


class NonBottleNeck(nn.Module):
    def __init__(self, inplanes, outplanes, dilated_rate):
        super(NonBottleNeck, self).__init__()
        self.conv1_v = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 1), 
                                 padding=(dilated_rate, 0), dilation=dilated_rate)
        self.conv1_h = nn.Conv2d(outplanes, outplanes, kernel_size=(1, 3), 
                                 padding=(0, dilated_rate), dilation=dilated_rate)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu1_v = nn.ReLU(inplace=True)
        self.relu1_h = nn.ReLU(inplace=True)

        self.conv2_v = nn.Conv2d(outplanes, outplanes, kernel_size=(3, 1),
                                 padding=(dilated_rate, 0), dilation=dilated_rate)
        self.conv2_h = nn.Conv2d(outplanes, outplanes, kernel_size=(1, 3),
                                 padding=(0, dilated_rate), dilation=dilated_rate)

        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu2_v = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_in = self.conv1_v(x)
        x_in = self.relu1_v(x_in)
        x_in = self.conv1_h(x_in)
        x_in = self.relu1_h(x_in)
        x_in = self.bn1(x_in)

        x_in = self.conv2_v(x_in)
        x_in = self.relu2_v(x_in)
        x_in = self.conv2_h(x_in)
        x_in = self.bn2(x_in)
        m = nn.Dropout2d(p=0.1)
        # x_in = F.dropout(x_in, p=0.30, training=self.training)
        x_in = m(x_in)
        out = x_in + x
        out = self.relu3(out)
        return out


class nxBottleNeck(nn.Module):
    def __init__(self, planes, times):
        super(nxBottleNeck, self).__init__()
        self.times = times
        self.bottleneck1 = NonBottleNeck(planes, planes, 1)
        self.bottleneck2 = NonBottleNeck(planes, planes, 1)
        self.bottleneck3 = NonBottleNeck(planes, planes, 1)
        self.bottleneck4 = NonBottleNeck(planes, planes, 1)
        self.bottleneck5 = NonBottleNeck(planes, planes, 1)
        self.bottleneck_list = nn.ModuleList()
        self.bottleneck_list.append(self.bottleneck1)
        self.bottleneck_list.append(self.bottleneck2)
        self.bottleneck_list.append(self.bottleneck3)
        self.bottleneck_list.append(self.bottleneck4)
        self.bottleneck_list.append(self.bottleneck5)

    def forward(self, x):
        for i in range(self.times):
            x = self.bottleneck_list[i](x)

        return x


class CasBottleNeck(nn.Module):
    def __init__(self, planes):
        super(CasBottleNeck, self).__init__()
        self.bottleneck1 = NonBottleNeck(planes, planes, 2)
        self.bottleneck2 = NonBottleNeck(planes, planes, 4)
        self.bottleneck3 = NonBottleNeck(planes, planes, 8)
        self.bottleneck4 = NonBottleNeck(planes, planes, 16)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        return x
        

class ERFNet(nn.Module):
    def __init__(self, configer):
        super(ERFNet, self).__init__()
        self.num_classes = configer.get('network', 'out_channels')
        self.downsampler1 = Downsampler(3, 16)
        self.downsampler2 = Downsampler(16, 64)
        self.bottleneck_5 = nxBottleNeck(64, 5)
        self.downsampler3 = Downsampler(64, 128)

        self.casbottleneck1 = CasBottleNeck(128)
        self.casbottleneck2 = CasBottleNeck(128)

        self.conv_enc = nn.Conv2d(128, self.num_classes, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.bottleneck_21 = nxBottleNeck(64, 2)
        self.deconv2 = nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.bottleneck_22 = nxBottleNeck(16, 2)
        self.deconv3 = nn.ConvTranspose2d(16, self.num_classes, 2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        x = self.downsampler1(x)
        x = self.downsampler2(x)
        x = self.bottleneck_5(x)
        x = self.downsampler3(x)
        x = self.casbottleneck1(x)
        x = self.casbottleneck2(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.bottleneck_21(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.bottleneck_22(x)
        x = self.deconv3(x)
        return x

if __name__ == "__main__":
    model = ERFNet(20)
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True)
    print (model(image).size())
