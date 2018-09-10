#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# VGG300 SSD model


import torch.nn.functional as F
from torch import nn

from utils.layers.det.ssd_detection_layer import SSDDetectionLayer
from models.backbones.backbone_selector import BackboneSelector


DETECTOR_CONFIG = {
    'num_centrals': [256, 128, 128, 128, 128],
    'num_strides': [2, 2, 2, 2, 2],
    'num_padding': [1, 1, 1, 1, 1],
    'vgg_cfg': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
}


class Vgg512SSD(nn.Module):

    def __init__(self, configer):
        super(Vgg512SSD, self).__init__()
        self.vgg_features = BackboneSelector(configer).get_backbone(vgg_cfg=DETECTOR_CONFIG['vgg_cfg'])
        self.ssd_head = SSDHead(configer)

    def forward(self, x):
        x = self.vgg_features(x)
        out = self.ssd_head(x)
        return out


class SSDHead(nn.Module):

    def __init__(self, configer):
        super(SSDHead, self).__init__()

        self.configer = configer
        self.num_features = self.configer.get('network', 'num_feature_list')
        self.num_centrals = DETECTOR_CONFIG['num_centrals']
        self.num_paddings = DETECTOR_CONFIG['num_padding']
        self.num_strides = DETECTOR_CONFIG['num_strides']
        self.norm4 = L2Norm2d(20)
        self.feature1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, self.num_features[1], kernel_size=3, padding=6, dilation=6),
            nn.ReLU(),
            nn.Conv2d(self.num_features[1], self.num_features[1], kernel_size=1),
            nn.ReLU(),
        )

        # 'num_features': [512, 1024, 512, 256, 256, 256].
        # 'num_centrals': [256, 128, 128, 128],
        # 'num_strides': [2, 2, 1, 1],
        # 'num_padding': [1, 1, 0, 0],
        self.feature2 = self.__extra_layer(num_in=self.num_features[1], num_out=self.num_features[2],
                                           num_c=self.num_centrals[0], stride=self.num_strides[0],
                                           pad=self.num_paddings[0])
        self.feature3 = self.__extra_layer(num_in=self.num_features[2], num_out=self.num_features[3],
                                           num_c=self.num_centrals[1], stride=self.num_strides[1],
                                           pad=self.num_paddings[1])
        self.feature4 = self.__extra_layer(num_in=self.num_features[3], num_out=self.num_features[4],
                                           num_c=self.num_centrals[2], stride=self.num_strides[2],
                                           pad=self.num_paddings[2])
        self.feature5 = self.__extra_layer(num_in=self.num_features[4], num_out=self.num_features[5],
                                           num_c=self.num_centrals[3], stride=self.num_strides[3],
                                           pad=self.num_paddings[3])
        self.feature6 = self.__extra_layer(num_in=self.num_features[5], num_out=self.num_features[6],
                                           num_c=self.num_centrals[4], stride=self.num_strides[4],
                                           pad=self.num_paddings[4])

        self.ssd_detection_layer = SSDDetectionLayer(configer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def __extra_layer(num_in, num_out, num_c, stride, pad):
        layer = nn.Sequential(
            nn.Conv2d(num_in, num_c, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_c, num_out, kernel_size=3, stride=stride, padding=pad),
            nn.ReLU(),
        )
        return layer

    def forward(self, feature):
        det_feature = list()
        det_feature.append(self.norm4(feature))
        feature = F.max_pool2d(feature, kernel_size=2, stride=2, ceil_mode=True)

        feature = self.feature1(feature)
        det_feature.append(feature)

        feature = self.feature2(feature)
        det_feature.append(feature)

        feature = self.feature3(feature)
        det_feature.append(feature)

        feature = self.feature4(feature)
        det_feature.append(feature)

        feature = self.feature5(feature)
        det_feature.append(feature)

        feature = self.feature6(feature)
        det_feature.append(feature)

        loc_preds, conf_preds = self.ssd_detection_layer(det_feature)

        return det_feature, loc_preds, conf_preds


class L2Norm2d(nn.Module):
    """L2Norm layer across all channels."""

    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        """out = scale * x / sqrt(\sum x_i^2)"""

        _sum = x.pow(2).sum(dim).clamp(min=1e-12).rsqrt()
        out = self.scale * x * _sum.unsqueeze(1).expand_as(x)
        return out


if __name__ == "__main__":
    pass

