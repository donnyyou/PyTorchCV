#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: YangMaoke, DuanZhixiang({maokeyang, zhixiangduan}@deepmotion.ai)
# The class of DenseASPPDetecNet


import torch
import torch.nn.init as init
from torch import nn

from models.backbones.backbone_selector import BackboneSelector
from utils.layers.det.ssd_multibox_layer import SSDShareMultiBoxLayer


class _DetFeatureBlock(nn.Sequential):
    def __init__(self, input_num, out_num, is_pooling=True):
        super(_DetFeatureBlock, self).__init__()
        if is_pooling:
            self.add_module('pool1', nn.AvgPool2d(kernel_size=2, stride=2))

        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=2*input_num, kernel_size=3,
                                           padding=1)),

        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(in_channels=2*input_num, out_channels=out_num, kernel_size=1))

    def forward(self, _input):
        feature = super(_DetFeatureBlock, self).forward(_input)
        return feature


class _RevertedResDetBlock(nn.Sequential):
    def __init__(self, input_num, out_num, stride, expand_ratio):
        super(_RevertedResDetBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_num, input_num*expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_num*expand_ratio),
            nn.ReLU6(inplace=False),

            nn.Conv2d(input_num*expand_ratio, input_num*expand_ratio, 3, stride, 1, groups=input_num*expand_ratio, bias=False),
            nn.BatchNorm2d(input_num*expand_ratio),
            nn.ReLU6(inplace=False),

            nn.Conv2d(input_num*expand_ratio, out_num, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_num),

        )

    def forward(self, x):
        feature = super(_RevertedResDetBlock, self).forward(x)
        return feature


class DenseASPPSSD(nn.Module):
    def __init__(self, configer):
        super(DenseASPPSSD, self).__init__()

        self.configer = configer
        det_features = self.configer.get('details', 'num_feature_list')
        self.num_classes = self.configer.get('data', 'num_classes')

        self.features = BackboneSelector(configer).get_backbone()

        num_features = self.features.get_num_features()

        self.ASPP_3 = nn.Sequential(nn.ReLU(inplace=False),
                                    nn.Conv2d(in_channels=num_features, out_channels=256, kernel_size=1),
                                    nn.BatchNorm2d(num_features=256),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, dilation=3, padding=3),
                                    nn.Dropout2d(p=0.1))

        self.ASPP_6 = nn.Sequential(nn.BatchNorm2d(num_features=num_features + 64 * 1),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(in_channels=num_features + 64 * 1, out_channels=256, kernel_size=1),

                                    nn.BatchNorm2d(num_features=256),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, dilation=6, padding=6),
                                    nn.Dropout2d(p=0.1))

        self.ASPP_12 = nn.Sequential(nn.BatchNorm2d(num_features=num_features + 64 * 2),
                                     nn.ReLU(inplace=False),
                                     nn.Conv2d(in_channels=num_features + 64 * 2, out_channels=256, kernel_size=1),

                                     nn.BatchNorm2d(num_features=256),
                                     nn.ReLU(inplace=False),
                                     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, dilation=12,
                                               padding=12),
                                     nn.Dropout2d(p=0.1))

        self.ASPP_18 = nn.Sequential(nn.BatchNorm2d(num_features=num_features + 64 * 3),
                                     nn.ReLU(inplace=False),
                                     nn.Conv2d(in_channels=num_features + 64 * 3, out_channels=256, kernel_size=1),

                                     nn.BatchNorm2d(num_features=256),
                                     nn.ReLU(inplace=False),
                                     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, dilation=18,
                                               padding=18),
                                     nn.Dropout2d(p=0.1))

        self.ASPP_24 = nn.Sequential(nn.BatchNorm2d(num_features=num_features + 64 * 4),
                                     nn.ReLU(inplace=False),
                                     nn.Conv2d(in_channels=num_features + 64 * 4, out_channels=256, kernel_size=1),
                                     nn.BatchNorm2d(num_features=256),
                                     nn.ReLU(inplace=False),
                                     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, dilation=24, padding=24))

        num_features += 5 * 64

        self.det_feature1 = _RevertedResDetBlock(num_features, det_features[0], stride=1, expand_ratio=2)
        self.det_feature2 = _RevertedResDetBlock(det_features[0], det_features[1], stride=2, expand_ratio=3)
        self.det_feature3 = _RevertedResDetBlock(det_features[1], det_features[2], stride=2, expand_ratio=3)
        self.det_feature4 = _RevertedResDetBlock(det_features[2], det_features[3], stride=2, expand_ratio=3)
        self.det_feature5 = _RevertedResDetBlock(det_features[3], det_features[4], stride=2, expand_ratio=3)

        self.multibox_layer = SSDShareMultiBoxLayer(configer)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, _input):
        feature = self.features(_input)

        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        det_feature = []
        feature = self.det_feature1(feature)
        det_feature.append(feature)
        feature = self.det_feature2(feature)
        det_feature.append(feature)
        feature = self.det_feature3(feature)
        det_feature.append(feature)
        feature = self.det_feature4(feature)
        det_feature.append(feature)
        feature = self.det_feature5(feature)
        det_feature.append(feature)

        loc_preds, conf_preds = self.multibox_layer(det_feature)
        return loc_preds, conf_preds

    @staticmethod
    def __extra_layer(num_in, num_out, num_c, stride, pad):
        layer = nn.Sequential(
            nn.Conv2d(num_in, num_c, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_c, num_out, kernel_size=3, stride=stride, padding=pad),
            nn.ReLU(),
        )
        return layer

    def load_pretrained_weight(self, net):
        self.features.load_pretrained_weight(net)


if __name__ == '__main__':
    DETECTOR_CONFIG = {
        'model_name': 'DenseASPPDet',
        'num_classes': 2,
        'num_anchors': [6, 6, 6, 6, 6],
        'num_centrals': [128, 128, 128, 128, 128],
        'num_strides': [2, 2, 2, 2, 2],
        'num_padding': [1, 1, 1, 1, 1],
        'base_name': 'squeezenet',
        'model_path': '../pretrained/squeezenet1_1.pth',
        'num_features': [256, 256, 256, 256, 256],
    }
