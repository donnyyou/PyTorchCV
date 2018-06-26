#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from torch.utils import data

from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class FSDataLoader(data.Dataset):
    def __init__(self, root_dir, aug_transform=None,
                 img_transform=None, label_transform=None, configer=None):
        self.img_list, self.label_list = self.__list_dirs(root_dir)
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.pil_open_rgb(self.img_list[index])
        label = ImageHelper.pil_open_p(self.label_list[index])

        if self.aug_transform is not None:
            img, label = self.aug_transform(img, label=label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __list_dirs(self, root_dir):
        img_list = list()
        label_list = list()
        image_dir = os.path.join(root_dir, 'image')
        label_dir = os.path.join(root_dir, 'label')
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        for file_name in os.listdir(label_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            img_list.append(os.path.join(image_dir, '{}.{}'.format(image_name, img_extension)))
            label_path = os.path.join(label_dir, file_name)
            label_list.append(label_path)
            if not os.path.exists(label_path):
                Log.error('Label Path: {} not exists.'.format(label_path))
                exit(1)

        return img_list, label_list


if __name__ == "__main__":
    # Test cityscapes loader.
    pass