#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Single Shot Detector data loader


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
import torch.utils.data as data

from extensions.parallel.data_container import DataContainer
from utils.helpers.json_helper import JsonHelper
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class YOLODataLoader(data.Dataset):

    def __init__(self, root_dir=None, aug_transform=None,
                 img_transform=None, configer=None):
        super(YOLODataLoader, self).__init__()
        self.img_list, self.json_list = self.__list_dirs(root_dir)
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))

        bboxes, labels = self.__read_json_file(self.json_list[index])

        if self.aug_transform is not None:
            img, bboxes, labels = self.aug_transform(img, bboxes=bboxes, labels=labels)

        labels = torch.from_numpy(labels).long()
        bboxes = torch.from_numpy(bboxes).float()

        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            img=DataContainer(img, stack=True),
            bboxes=DataContainer(bboxes, stack=False),
            labels=DataContainer(labels, stack=False),
        )

    def __len__(self):

        return len(self.img_list)

    def __read_json_file(self, json_file):
        """
            filename: JSON file

            return: three list: key_points list, centers list and scales list.
        """
        json_dict = JsonHelper.load_file(json_file)

        labels = list()
        bboxes = list()

        for object in json_dict['objects']:
            if 'difficult' in object and object['difficult'] and not self.configer.get('data', 'keep_difficult'):
                continue

            labels.append(object['label'])
            bboxes.append(object['bbox'])

        return np.array(bboxes).astype(np.float32), np.array(labels)

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