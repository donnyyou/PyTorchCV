"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import collections.abc

import numpy as np
import torch
import torch.nn as nn
import torch.autograd

import model


settings = {
    'size': (416, 416),
}


def reorg(x, stride_h=2, stride_w=2):
    batch_size, channels, height, width = x.size()
    _height, _width = height // stride_h, width // stride_w
    if 1:
        x = x.view(batch_size, channels, _height, stride_h, _width, stride_w).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, stride_h * stride_w).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, stride_h * stride_w, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)
    else:
        x = x.view(batch_size, channels, _height, stride_h, _width, stride_w)
        x = x.permute(0, 1, 3, 5, 2, 4) # batch_size, channels, stride, stride, _height, _width
        x = x.contiguous()
        x = x.view(batch_size, -1, _height, _width)
    return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bn=True, act=True):
        nn.Module.__init__(self)
        if isinstance(padding, bool):
            if isinstance(kernel_size, collections.abc.Iterable):
                padding = tuple((kernel_size - 1) // 2 for kernel_size in kernel_size) if padding else 0
            else:
                padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else lambda x: x
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Darknet(nn.Module):
    def __init__(self, config_channels, anchors, num_cls, stride=2, ratio=1):
        nn.Module.__init__(self)
        self.stride = stride
        channels = int(32 * ratio)
        layers = []

        bn = config_channels.config.getboolean('batch_norm', 'enable')
        # layers1
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            channels *= 2
        # down 4
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
            layers.append(Conv2d(config_channels.channels, config_channels(channels // 2, 'layers1.%d.conv.weight' % len(layers)), 1, bn=bn))
            layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            channels *= 2
        # down 16
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
            layers.append(Conv2d(config_channels.channels, config_channels(channels // 2, 'layers1.%d.conv.weight' % len(layers)), 1, bn=bn))
        layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
        self.layers1 = nn.Sequential(*layers)

        # layers2
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=2))
        channels *= 2
        # down 32
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers2.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
            layers.append(Conv2d(config_channels.channels, config_channels(channels // 2, 'layers2.%d.conv.weight' % len(layers)), 1, bn=bn))
        for _ in range(3):
            layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers2.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
        self.layers2 = nn.Sequential(*layers)

        self.passthrough = Conv2d(self.layers1[-1].conv.weight.size(0), config_channels(int(64 * ratio), 'passthrough.conv.weight'), 1, bn=bn)

        # layers3
        layers = []
        layers.append(Conv2d(self.passthrough.conv.weight.size(0) * self.stride * self.stride + self.layers2[-1].conv.weight.size(0), config_channels(int(1024 * ratio), 'layers3.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
        layers.append(Conv2d(config_channels.channels, model.output_channels(len(anchors), num_cls), 1, bn=False, act=False))
        self.layers3 = nn.Sequential(*layers)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layers1(x)
        _x = reorg(self.passthrough(x), self.stride)
        x = self.layers2(x)
        x = torch.cat([_x, x], 1)
        return self.layers3(x)

    def scope(self, name):
        return '.'.join(name.split('.')[:-2])

    def get_mapper(self, index):
        if index == 94:
            return lambda indices, channels: torch.cat([indices + i * channels for i in range(self.stride * self.stride)])


class Tiny(nn.Module):
    def __init__(self, config_channels, anchors, num_cls, channels=16):
        nn.Module.__init__(self)
        layers = []

        bn = config_channels.config.getboolean('batch_norm', 'enable')
        for _ in range(5):
            layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            channels *= 2
        layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
        layers.append(nn.ConstantPad2d((0, 1, 0, 1), float(np.finfo(np.float32).min)))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
        channels *= 2
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(channels, 'layers.%d.conv.weight' % len(layers)), 3, bn=bn, padding=True))
        layers.append(Conv2d(config_channels.channels, model.output_channels(len(anchors), num_cls), 1, bn=False, act=False))
        self.layers = nn.Sequential(*layers)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)

    def scope(self, name):
        return '.'.join(name.split('.')[:-2])