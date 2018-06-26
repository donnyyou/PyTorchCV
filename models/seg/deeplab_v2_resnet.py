#!/usr/bin/env python
# -*- coding:utf-8 -*-
# deeplabv2 res101
# Author: Xiangtai(lxtpku@pku.edu.cn)


import torch.nn as nn
import torch.nn.functional as F

from deeplab_resnet_synbn import ModelBuilder
from extensions.layers.nn import SynchronizedBatchNorm2d


class _ASPPModule(nn.Module):
	"""
	Atrous Spatial Pyramid Pooling module
	"""
	def __init__(self, inplanes, outplanes, pyramids):
		super(_ASPPModule, self).__init__()
		self.model = nn.Module()
		for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
			# note same dilation with the same padding
			submodel = nn.Sequential(
				nn.Conv2d(
					in_channels=inplanes,
					out_channels=inplanes,
					kernel_size=3,
					stride=1,
					padding=padding,
					dilation=dilation,
					bias=True
				),
				nn.Conv2d(
					in_channels=inplanes,
					out_channels=inplanes,
					kernel_size=1,
					stride=1
				),
				nn.Conv2d(
					in_channels=inplanes,
					out_channels=outplanes,
					kernel_size=1,
					stride=1
				)
			)
			self.model.add_module(
				'aspp_{}'.format(i),
				submodel
			)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal(m.weight, mean=0, std=0.01)
				nn.init.constant(m.bias, 0)

	def forward(self, x):
		res = 0
		for stage in self.model.children():
			res += stage(x)
		return res


class DeepLabV2(nn.Module):
	"""Deeplab V2"""
	def __init__(self, n_classes, small=True, decoder=None):
		super(DeepLabV2, self).__init__()
		self.resnet_features = ModelBuilder().build_encoder("resnet101_dilated8")
		if small:
			pyramids = [2, 4, 8, 12]
		else:
			pyramids = [6, 12, 18, 24]
		self.aspp = _ASPPModule(2048, n_classes, pyramids)
		self.decoder = decoder

	def forward(self, x):
		x = self.resnet_features(x)
		x = self.aspp(x)

		if self.decoder == None:
			x = F.upsample(x, scale_factor=8, mode="bilinear")
		else:
			x = self.decoder(x)
		return x

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()
			if isinstance(m, SynchronizedBatchNorm2d):
				m.eval()


# test the model
if __name__ == '__main__':
	model = DeepLabV2(n_classes=20)
	model.freeze_bn()
	model.eval()
	print (model.resnet_features.layer4)