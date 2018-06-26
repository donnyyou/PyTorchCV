#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai Li(xiangtai94@gmail.com)
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch.autograd import Variable

from extensions.flow import Correlation
from flow_misc import conv, deconv, predict_flow, tofp16, tofp32

'Parameter count , 39,175,298 '


class FlowNetC(nn.Module):

    def __init__(self, with_bn=True, fp16=False):
        super(FlowNetC, self).__init__()

        self.with_bn = with_bn
        self.fp16 = fp16

        self.conv1 = conv(3, 64, kernel_size=7, stride=2, with_bn=with_bn)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv_redir = conv(
            256, 32, kernel_size=1, stride=1, with_bn=with_bn)

        corr = Correlation(
            pad_size=20,
            kernel_size=1,
            max_displacement=20,
            stride1=1,
            stride2=2,
            corr_multiply=1)
        self.corr = nn.Sequential(tofp32(), corr, tofp16()) if fp16 else corr

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(473, 256, with_bn=with_bn)
        self.conv4 = conv(256, 512, stride=2, with_bn=with_bn)
        self.conv4_1 = conv(512, 512, with_bn=with_bn)
        self.conv5 = conv(512, 512, stride=2, with_bn=with_bn)
        self.conv5_1 = conv(512, 512, with_bn=with_bn)
        self.conv6 = conv(512, 1024, stride=2, with_bn=with_bn)
        self.conv6_1 = conv(1024, 1024, with_bn=with_bn)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        # print (out_conv3a.size(), out_conv3b.size())

        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)
        # print (out_corr.size())
        # exit()
        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)

        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)

        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)

        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,

if __name__ == '__main__':
    net2 = FlowNetC().cuda()
    img = torch.rand(1, 6, 512, 1024)
    # print (img)
    out = net2(Variable(img).cuda())
    print (out)