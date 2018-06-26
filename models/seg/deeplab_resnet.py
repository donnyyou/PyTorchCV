#!/usr/bin/env python
# coding: utf-8
# Author: Xiangtai Li
# some helper functions used in the deeplab resnet models

import torch.nn as nn
import torch.nn.functional as F


class _ConvBatchNormReluBlock(nn.Sequential):
	def __init__(self, inplanes, outplanes, kernel_size, stride, padding, dilation, relu=True):
		super(_ConvBatchNormReluBlock, self).__init__()
		self.add_module("cov", nn.Conv2d(in_channels=inplanes,out_channels=outplanes,
							kernel_size=kernel_size, stride=stride, padding = padding,
							dilation = dilation, bias=False))
		self.add_module("bn", nn.BatchNorm2d(num_features=outplanes, momentum=0.999, affine=True))
		if relu:
			self.add_module("relu", nn.ReLU())
	def forward(self, x):
		return super(_ConvBatchNormReluBlock, self).forward(x)

class _Bottleneck(nn.Sequential):
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

class _ResidualBlock(nn.Sequential):
	"""
		Residual Block with dilation setting
	"""
	def __init__(self, layers, inplanes, midplanes, outplanes, stride, dilation):
		super(_ResidualBlock, self).__init__()
		self.add_module("block1", _Bottleneck(inplanes, midplanes, outplanes,stride,dilation,True))
		for i in range(2, layers+1):
			self.add_module("block"+str(i), _Bottleneck(outplanes,midplanes, outplanes,1,dilation,False))
	def forward(self, x):
		return super(_ResidualBlock, self).forward(x)

class _ResidualBlockMulGrid(nn.Sequential):
	"""
		Residual Block with multi-grid
	"""
	def __init__(self, layers, inplanes, midplanes, outplanes, stride, dilation, mulgrid=[1,2,1]):
		super(_ResidualBlockMulGrid, self).__init__()
		self.add_module("block1", _Bottleneck(inplanes, midplanes, outplanes, stride, dilation * mulgrid[0], True))
		self.add_module("block2", _Bottleneck(outplanes, midplanes, outplanes, stride, dilation * mulgrid[1], False))
		self.add_module("block3", _Bottleneck(outplanes, midplanes, outplanes, stride, dilation * mulgrid[2], False))
	def forward(self, x):
		return super(_ResidualBlockMulGrid, self).forward(x)