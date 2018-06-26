#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai(lxtpku@pku.edu.cn)
# Pytorch implementation of paper Low-Latency Video Semantic Segmentation (CVPR2018)
# Test Version
"""
    Key idea: low level features matter
    baseline: resnet-101 (here use the deeplab v3+)
    training methods:
    1, first train on the cityscape use the baseline network.(fixed as feature extractor)
    2, fine tune the adaptive propagation module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.layers.nn import SynchronizedBatchNorm2d
from models.seg.deeplab_resnet_synbn import _ResidualBlockMulGrid, ModelBuilder
from models.seg.deeplab_v3_resnet import _ASPPModule


class WeightPredictor(nn.Module):

    def __init__(self, in_channels):
        super(WeightPredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Conv2d(256, 81, kernel_size=1, stride=1)  # 9 * 9 kernel

    def forward(self, x):
        f1, f2 = x
        f = torch.cat([self.conv1(f1), self.conv1(f2)], dim=1)
        f = self.conv2(f)
        out = F.softmax(self.fc(f), dim=1)
        return out


class AdaptiveScheduler(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveScheduler, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1 ,padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)  # output a deviation

    def forward(self, x):
        """
        :param x: f1,f2 (low level features)
        :return:
        """
        f1, f2 = x
        diff = self.conv1(f1) - self.conv1(f2)
        out = self.conv2(diff)
        print (out.size())
        out = self.pool(out).squeeze()
        print (out.size())
        out = self.fc(out)
        print (out.size())
        return out


class PropagationModule(nn.Module):

    def __init__(self, kernel=9):
        super(PropagationModule, self).__init__()
        self.kernel = kernel

    def forward(self, x):
        key_feature, weights = x
        n, c, h, w = key_feature.size()
        print ("key feature", key_feature.size())
        img2col = self.img2col(key_feature)
        print ("img2col", img2col.size())
        weights = weights.squeeze().contiguous().view(self.kernel**2, -1).permute(1, 0)
        print ("weights:", weights.size())
        res = img2col * weights
        print ("res: ", res.size())
        return res.sum(dim=3).view(n, c, h, w)
        """
        # low speed version
        tmp = []
        for i in range(c):
            img2col_channel = img2col[:, i, :, :].squeeze().view(n*h*w, -1)
            res_tmp = (img2col_channel * weights).sum(dim=1)
            res_tmp = res_tmp.unsqueeze(1).view(-1,1,h,w)
            tmp.append(res_tmp)
        res = torch.cat(tmp, dim=1)
        return res
        """

    def img2col(self, x):
        n, c, w, h = x.size()
        feature_pad = F.pad(x, (self.kernel//2, self.kernel//2, self.kernel//2,self.kernel//2), "constant", value=0)
        img2col = feature_pad.unfold(2, self.kernel, 1).unfold(3, self.kernel, 1).contiguous().view(n, c, w*h, self.kernel**2)
        return img2col



class AdaptNet(nn.Module):

    def __init__(self, in_channels):
        super(AdaptNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1, x2 = x
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LowLatencyModel(nn.Module):
    def __init__(self, num_classes, pyramids=[6, 12, 18], multi_grid=[1, 2, 4], threshold=0.30):
        super(LowLatencyModel, self).__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.resnet_features = ModelBuilder().build_encoder("resnet101")
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
                                 SynchronizedBatchNorm2d(256))

        self.reduce_conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1),
                                          SynchronizedBatchNorm2d(48))
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
                                       SynchronizedBatchNorm2d(256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       SynchronizedBatchNorm2d(256),
                                       nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1))

        self.adaptNet = AdaptNet(512)
        self.scheduler = AdaptiveScheduler(256)
        self.weightPredictor = WeightPredictor(256)
        self.progation = PropagationModule(kernel=9)

    def trainsingle(self, x):
        low = self.low_features(x)
        print ("low feature: ", low.size())
        x = self.high_features(low)
        x = self.MG_features(x)
        print (x.size())
        x = self.aspp(x)
        x = self.fc1(x)
        x = F.upsample(x, scale_factor=(4, 4), mode='bilinear')
        print ("deep feature: ", x.size())
        low = self.reduce_conv2(low)

        x = torch.cat((x, low), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=(4, 4), mode='bilinear')

        return x

    def forward(self, x):
        """
        this method is used for training frame pair
        :param x: two frame pairs
        :return:
        """
        f1, f2 = x  # first frame is key frame, the second frame contains ground truth
        f1_deep, f1_low= self.getDeepfeature(f1)
        f2_low = self.getLowfeature(f2)
        print ("f1 f2 low",f1_low.size(), f2_low.size())

        kernel = self.weightPredictor([f1_low, f2_low])
        deviation = self.scheduler([f1_low, f2_low])

        progated_feature = self.progation([f1_deep, kernel])
        print ("progated ", progated_feature.size())
        fuse_feature = self.adaptNet([f2_low, progated_feature])

        print ("fuse feature", fuse_feature.size())
        # this is for decoder part
        low = self.reduce_conv2(f2_low)

        x = torch.cat((fuse_feature, low), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=(4, 4), mode='bilinear')

        return x, deviation

    def inference(self, x, isFirst):
        """
        this method is used for video sequence inference
        :param x:
        :param isFirst:
        :return:
        """
        pass

    def getDeepfeature(self, x):
        low = self.low_features(x)
        x = self.high_features(low)
        x = self.MG_features(x)
        x = self.aspp(x)
        x = self.fc1(x)
        x = F.upsample(x, scale_factor=(4, 4), mode='bilinear')

        return x, low

    def getLowfeature(self, x):
        return self.low_features(x)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()

if __name__ == '__main__':
    model = LowLatencyModel(19, pyramids=[6, 12, 18]).cuda()
    model.freeze_bn()
    model.eval()

    f1 = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True).cuda()
    f2 = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True).cuda()
    print (model([f1,f2])[0].size())