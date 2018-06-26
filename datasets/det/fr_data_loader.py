#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiantai li
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import torch
import torch.utils.data as data
from torchvision.transforms import ToTensor
import cv2

from utils.tools.logger import Logger as Log


class FRDataLoader(data.Dataset):
	def __init__(self, root_dir=None, aug_transform=None,
				 img_transform=None, configer=None):
		super(FRDataLoader, self).__init__()
		self.img_list, self.json_list = self.__list_dirs(root_dir,mode)
		self.aug_transform = aug_transform
		self.img_transform = img_transform
		self.boxes_max = 20

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, index):
		img = cv2.imread(self.img_list[index])
		img = cv2.resize(img,(600, 900))

		labels, bboxes = self.__read_json_file(self.json_list[index])

		if self.aug_transform is not None:
			img, bboxes = self.aug_transform(img, bboxes=bboxes)

		if self.img_transform is not None:
			img = self.img_transform(img)

		img_info = torch.FloatTensor([600, 900])

		labels, bboxes= self.__toTensor(labels), self.__toTensor(bboxes)

		boxes_num = bboxes.size(0)
		gtboxes = self.__fill_box_target(bboxes,labels,boxes_num,self.boxes_max)

		return img, img_info, gtboxes, boxes_num


	def __read_json_file(self, json_file):
		"""
		read annotation from json file
		:param json_file:
		:return:
		"""
		with open(json_file, 'r') as fp:
			node = json.load(fp)
			labels = list()
			bboxes = list()
			for object in node['objects']:
				labels.append(object['label'])
				bboxes.append(object['bbox'])

		return labels, bboxes

	def __list_dirs(self, root_dir):
		img_list = list()
		json_list = list()
		image_dir = os.path.join(root_dir, 'image')
		json_dir = os.path.join(root_dir, 'json')

		img_extension = os.listdir(image_dir)[0].split('.')[-1]

		for file_name in os.listdir(json_dir):
			image_name = '.'.join(file_name.split('.')[:-1])
			img_list.append(os.path.join(image_dir, '{}.{}'.format(image_name, img_extension)))
			json_path = os.path.join(json_dir, file_name)
			json_list.append(json_path)
			if not os.path.exists(json_path):
				Log.error('Json Path: {} not exists.'.format(json_path))
				exit(1)

		return img_list, json_list

	def __fill_box_target(self, boxes, labels, num, maxnum):
		gtboxes = torch.FloatTensor(maxnum, 5).fill_(0)
		gtboxes[:num,:4] = boxes
		gtboxes[:num, 4] = labels
		return gtboxes

	def __toTensor(self, array):
		return torch.FloatTensor(array)


if __name__ == '__main__':

	im_data = torch.FloatTensor(1)
	im_info = torch.FloatTensor(1)
	num_boxes = torch.LongTensor(1)
	gt_boxes = torch.FloatTensor(1)

	dataSet = FRDataLoader(root_dir="/home/lxt/data/VOC2007", img_transform=ToTensor())
	dataLoader = data.DataLoader(dataSet, batch_size=1, num_workers=8)

	for i, data in enumerate(dataLoader):
		if i == 1 :
			break
		im_data.resize_(data[0].size()).copy_(data[0])
		im_info.resize_(data[1].size()).copy_(data[1])
		gt_boxes.resize_(data[2].size()).copy_(data[2])
		num_boxes.resize_(data[3].size()).copy_(data[3])

		print("image data size:", im_data.size())  # (n,c,h,w)
		print("image_info size", im_info.size())  # (n,3)
		print("image_info content", im_info)
		print("ground truth boxes size", gt_boxes.size())  # (n, 20, 5)
		print("num_boxes size", num_boxes.size())  # (1L,)
		print("num_boexs content", num_boxes)