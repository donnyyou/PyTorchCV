#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# OpenPose data loader for keypoints detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch.utils.data as data
from PIL import Image
from utils.helpers.json_helper import JsonHelper

from datasets.pose.pose_data_utilizer import PoseDataUtilizer
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class CPDataLoader(data.Dataset):

    def __init__(self, root_dir = None, aug_transform=None,
                 img_transform=None, label_transform=None, configer=None):

        self.img_list, self.json_list, self.mask_list = self.__list_dirs(root_dir)
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.pose_data_utilizer = PoseDataUtilizer(configer)

    def __getitem__(self, index):
        img = ImageHelper.pil_open_rgb(self.img_list[index])
        if os.path.exists(self.mask_list[index]):
            maskmap = ImageHelper.pil_open_p(self.mask_list[index])
        else:
            maskmap = ImageHelper.np2img(np.ones((img.size[1], img.size[0]), dtype=np.uint8))

        kpts, bboxes = self.__read_json_file(self.json_list[index])

        if self.aug_transform is not None and len(bboxes) > 0:
            img, maskmap, kpts, bboxes = self.aug_transform(img, mask=maskmap, kpts=kpts, bboxes=bboxes)

        elif self.aug_transform is not None:
            img, maskmap, kpts = self.aug_transform(img, mask=maskmap, kpts=kpts)

        width, height = maskmap.size
        maskmap = ImageHelper.resize(maskmap, (width // self.configer.get('network', 'stride'),
                                               height // self.configer.get('network', 'stride')), Image.CUBIC)

        maskmap = np.expand_dims(np.array(maskmap, dtype=np.float32), axis=2)

        partmap = self.pose_data_utilizer.generate_partmap(kpts=kpts, mask=maskmap)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            partmap = self.label_transform(partmap)
            maskmap = self.label_transform(maskmap)

        partmap = partmap.view(-1, 6, partmap.size(1), partmap.size(2))
        vecmap = partmap[:, 0:2, :, :].contiguous().view(-1, partmap.size(2), partmap.size(3))
        partmap = partmap[:, 2:6, :, :].contiguous().view(-1, partmap.size(2), partmap.size(3))
        return img, partmap, maskmap, vecmap

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
            kpts.append(object['keypoints'])
            if 'bbox' in object:
                bboxes.append(object['bbox'])

        return kpts, bboxes

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