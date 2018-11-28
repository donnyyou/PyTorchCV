#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# OpenPose data loader for keypoints detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
import torch.utils.data as data

from extensions.parallel.data_container import DataContainer
from utils.layers.pose.heatmap_generator import HeatmapGenerator
from utils.layers.pose.paf_generator import PafGenerator
from utils.helpers.json_helper import JsonHelper
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class OPDataLoader(data.Dataset):

    def __init__(self, root_dir = None, aug_transform=None,
                 img_transform=None, configer=None):

        self.img_list, self.json_list, self.mask_list = self.__list_dirs(root_dir)
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.heatmap_generator = HeatmapGenerator(self.configer)
        self.paf_generator = PafGenerator(self.configer)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        if os.path.exists(self.mask_list[index]):
            maskmap = ImageHelper.read_image(self.mask_list[index],
                                             tool=self.configer.get('data', 'image_tool'), mode='P')
        else:
            maskmap = np.ones((img.size[1], img.size[0]), dtype=np.uint8)
            if self.configer.get('data', 'image_tool') == 'pil':
                maskmap = ImageHelper.np2img(maskmap)

        kpts, bboxes = self.__read_json_file(self.json_list[index])

        if self.aug_transform is not None and len(bboxes) > 0:
            img, maskmap, kpts, bboxes = self.aug_transform(img, maskmap=maskmap, kpts=kpts, bboxes=bboxes)

        elif self.aug_transform is not None:
            img, maskmap, kpts = self.aug_transform(img, maskmap=maskmap, kpts=kpts)

        width, height = ImageHelper.get_size(maskmap)
        maskmap = ImageHelper.resize(maskmap,
                                     (width // self.configer.get('network', 'stride'),
                                      height // self.configer.get('network', 'stride')),
                                     interpolation='nearest')

        maskmap = torch.from_numpy(np.array(maskmap, dtype=np.float32))
        maskmap = maskmap.unsqueeze(0)
        kpts = torch.from_numpy(kpts).float()

        heatmap = self.heatmap_generator(kpts, [width, height], maskmap)
        vecmap = self.paf_generator(kpts, [width, height], maskmap)
        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            img=DataContainer(img, stack=True),
            heatmap=DataContainer(heatmap, stack=True),
            maskmap=DataContainer(maskmap, stack=True),
            vecmap=DataContainer(vecmap, stack=True)
        )

    def __len__(self):

        return len(self.img_list)

    def __read_json_file(self, json_file):
        """
            filename: JSON file

            return: three list: key_points list, centers list and scales list.
        """
        json_dict = JsonHelper.load_file(json_file)

        kpts = list()
        bboxes = list()

        for object in json_dict['objects']:
            kpts.append(object['kpts'])
            if 'bbox' in object:
                bboxes.append(object['bbox'])

        return np.array(kpts).astype(np.float32), np.array(bboxes).astype(np.float32)

    def __list_dirs(self, root_dir):
        img_list = list()
        json_list = list()
        mask_list = list()
        image_dir = os.path.join(root_dir, 'image')
        json_dir = os.path.join(root_dir, 'json')
        mask_dir = os.path.join(root_dir, 'mask')

        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        for file_name in os.listdir(json_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            img_list.append(os.path.join(image_dir, '{}.{}'.format(image_name, img_extension)))
            mask_path = os.path.join(mask_dir, '{}.png'.format(image_name))
            mask_list.append(mask_path)
            json_path = os.path.join(json_dir, file_name)
            json_list.append(json_path)
            if not os.path.exists(json_path):
                Log.error('Json Path: {} not exists.'.format(json_path))
                exit(1)

        return img_list, json_list, mask_list


if __name__ == "__main__":
    # Test coco loader.
    pass
