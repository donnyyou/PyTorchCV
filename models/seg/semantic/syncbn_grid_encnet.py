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


class SyncBNGridEncNet(nn.Module):
    def __init__(self, configer):
        super(SyncBNGridEncNet, self).__init__()
        self.configer = configer
        self.backbone = BackboneSelector(configer).get_backbone()
        self.aux_loss = 'aux_loss' in self.configer.get('network', 'loss_weights')
        self.se_loss = 'se_loss' in self.configer.get('network', 'loss_weights')
        self.head = PyramidEncHead(self.backbone.num_features, self.configer.get('data', 'num_classes'),
                                   enc_size=self.configer.get('network', 'enc_size'), se_loss=self.se_loss,
                                   lateral=self.configer.get('network', 'lateral'))
        if self.aux_loss:
            self.auxlayer = FCNHead(1024, self.configer.get('data', 'num_classes'))

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.backbone(x, is_tuple=True)

        x = list(self.head(*features))
        x[0] = F.upsample(x[0], imsize)
        if self.aux_loss:
            auxout = self.auxlayer(features[2])
            auxout = F.upsample(auxout, imsize)
            x.append(auxout)

        return tuple(x)


# PSP decoder Part
# pyramid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    def __init__(self, pool_scales=(1, 2, 3, 6), fc_dim=1024):
        super(PPMBilinearDeepsup, self).__init__()
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

    def forward(self, conv5):
        input_size = conv5.size()
        ppm_out = []

        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=True))

        ppm_out = torch.cat(ppm_out, 1)
        return ppm_out

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
        outputs = [F.relu(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))

        return tuple(outputs)


class PyramidEncHead(nn.Module):
    def __init__(self, in_channels, out_channels,
                 enc_size=4, pool_scales=(1, 2, 3, 6), se_loss=True, lateral=True):
        super(PyramidEncHead, self).__init__()
        self.enc_size = enc_size
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

        self.enc_module = EncModule(512, out_channels, ncodes=48, se_loss=se_loss)

        self.psp_module = PPMBilinearDeepsup(fc_dim=1024)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512 + 256 * len(pool_scales), out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))

        b, c, h, w = feat.size()
        pad_h = 0 if (h % self.enc_size == 0) else self.enc_size - (h % self.enc_size)
        pad_w = 0 if (w % self.enc_size == 0) else self.enc_size - (w % self.enc_size)
        feat = F.pad(feat, (0, pad_w, 0, pad_h), "constant", 0)
        b, c, h, w = feat.size()
        feat_temp = feat.contiguous().view(b, c, h // self.enc_size, self.enc_size, w // self.enc_size, self.enc_size)
        feat_temp = feat_temp.permute(0, 2, 4, 1, 3, 5).contiguous().view(b * h * w // (self.enc_size ** 2),
                                                                          c, self.enc_size, self.enc_size)
        se_outs = list(self.enc_module(feat_temp))
        feat_temp = se_outs[0].contiguous().view(b, h // self.enc_size,
                                                 w // self.enc_size, c, self.enc_size, self.enc_size)
        feat_temp = feat_temp.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, h, w)

        enc_out = feat_temp[:, :, :-pad_h, :-pad_w].contiguous()
        psp_out = self.psp_module(inputs[-1])
        out = self.conv6(torch.cat((enc_out, psp_out), 1))
        return out, se_outs[1]


if __name__ == "__main__":
    from utils.tools.configer import Configer
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    configer = Configer(
        hypes_file='/home/donny/Projects/PyTorchCV/hypes/seg/cityscape/fs_gridencnet_cityscape_seg.json')
    configer.add_key_value(['project_dir'], '/home/donny/Projects/PyTorchCV')
    model = SyncBNGridEncNet(configer).cuda()
    model.eval()
    image = torch.randn(1, 3, 96, 96).cuda()
    out = model(image)
    print(out[0].size())
