#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.layers.encoding.syncbn import BatchNorm1d, BatchNorm2d
from extensions.layers.encoding.encoding import Encoding
from models.tools.module_helper import Mean
from models.backbones.backbone_selector import BackboneSelector


class EncNet(nn.Module):
    def __init__(self, configer):
        super(EncNet, self).__init__()
        self.configer = configer
        self.backbone = BackboneSelector(configer).get_backbone()
        self.head = PyramidEncHead(2048, self.configer.get('data', 'num_classes'),
                                   pyramid=self.configer.get('network', 'pyramid'),
                                   se_loss=self.configer.get('network', 'se_loss'),
                                   lateral=self.configer.get('network', 'lateral'))
        self.aux = self.configer.get('network', 'aux')
        if self.aux:
            self.auxlayer = FCNHead(1024, self.configer.get('data', 'num_classes'))

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.backbone(x, is_tuple=True)

        x = list(self.head(*features))
        x[0] = F.upsample(x[0], imsize, **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.upsample(auxout, imsize, **self._up_kwargs)
            x.append(auxout)

        return tuple(x)


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True):
        super(EncModule, self).__init__()
        # norm_layer = nn.BatchNorm1d if isinstance(norm_layer, nn.BatchNorm2d) else \
        #    encoding.nn.BatchNorm1d
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Encoding(D=in_channels, K=ncodes),
            BatchNorm1d(ncodes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))

        return tuple(outputs)


class PyramidEncHead(nn.Module):
    def __init__(self, in_channels, out_channels, pyramid=[1, 2, 3, 6], se_loss=True, lateral=True):
        super(PyramidEncHead, self).__init__()
        self.pyramid = pyramid
        self.se_loss = se_loss
        self.lateral = lateral
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True))
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=1, bias=False),
                    BatchNorm2d(512),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                    BatchNorm2d(512),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                    BatchNorm2d(512),
                    nn.ReLU(inplace=True))
        self.encmodule_list = nn.ModuleList()
        for i in self.pyramid:
            self.encmodule_list.append(EncModule(512, out_channels, ncodes=48 // i, se_loss=se_loss))

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512*4, out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))

        b, c, h, w = feat.size()
        assert h % 6 == 0 and w % 6 == 0
        feat_list = list()
        se_list = list()
        for i, scale in enumerate(self.pyramid):
            feat_temp = feat.contiguous().view(b, c, h // scale, scale, w // scale, scale)
            feat_temp = feat_temp.permute(0, 3, 5, 1, 2, 4).contiguous().view(b*scale*scale, c, h // scale, w // scale)
            outs = list(self.encmodule_list[i](feat_temp))
            feat_temp = outs[0].contiguous().view(b, scale, scale, c, h // scale, w // scale)
            feat_temp = feat_temp.permute(0, 3, 4, 1, 5, 2).contiguous().view(b, c, h, w)
            feat_list.append(feat_temp)
            se_list.append(outs[1])

        out = self.conv6(torch.cat(feat_list, 1))

        return out, se_list