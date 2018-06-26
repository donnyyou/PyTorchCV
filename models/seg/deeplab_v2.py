#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai Li

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplab_resnet import _ConvBatchNormReluBlock, _ResidualBlock


class _ASPPModule(nn.Module):
	"""
	Atrous Spatial Pyramid Pooling module
	"""
	def __init__(self, inplanes, outplanes, pyramids):
		super(_ASPPModule, self).__init__()
		self.model = nn.Module()
		for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
			# note same dilation with the same padding
			self.model.add_module(
				'c{}'.format(i),
				nn.Conv2d(
					in_channels=inplanes,
					out_channels=outplanes,
					kernel_size=3,
					stride=1,
					padding=padding,
					dilation=dilation,
					bias=True
				)
			)
		for m in self.model.children():
			nn.init.normal(m.weight, mean=0, std=0.01)
			nn.init.constant(m.bias, 0)
	def forward(self, x):
		res = 0
		for stage in self.model.children():
			res += stage(x)
		return res
class DeepLabV2(nn.Sequential):
	"""Deep lab V2"""
	def __init__(self, n_classes, n_blocks, pyramids):
		super(DeepLabV2, self).__init__()
		self.add_module(
			'layer1',
			nn.Sequential(
				OrderedDict([
					('conv1', _ConvBatchNormReluBlock(3, 64, 7, 2, 3, 1)),
					('pool', nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
				])
			)
		)
		self.add_module('layer2', _ResidualBlock(n_blocks[0], 64, 64, 256, 1, 1))
		self.add_module('layer3', _ResidualBlock(n_blocks[1], 256, 128, 512, 2, 1))
		self.add_module('layer4', _ResidualBlock(n_blocks[2], 512, 256, 1024, 1, 2))
		self.add_module('layer5', _ResidualBlock(n_blocks[3], 1024, 512, 2048, 1, 4))
		self.add_module('aspp', _ASPPModule(2048, n_classes, pyramids))

	def forward(self, x):
		return super(DeepLabV2, self).forward(x)


	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()



if __name__ == '__main__':
    model = DeepLabV2(n_classes=21, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18, 24])
    model.freeze_bn()
    model.eval()
    print list(model.named_children())
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    print model.forward(image).size()