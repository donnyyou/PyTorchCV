#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from torch.utils import data

import datasets.tools.aug_transforms as aug_trans
import datasets.tools.transforms as trans
from datasets.det.ssd_data_loader import SSDDataLoader
from datasets.det.fr_data_loader import FRDataLoader
from utils.tools.logger import Logger as Log


class DetDataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        self.aug_train_transform = aug_trans.AugCompose(self.configer, split='train')

        self.aug_val_transform = aug_trans.AugCompose(self.configer, split='val')

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(mean=self.configer.get('trans_params', 'mean'),
                            std=self.configer.get('trans_params', 'std')), ])

    def get_trainloader(self):
        if self.configer.get('method') == 'single_shot_detector':
            trainloader = data.DataLoader(
                SSDDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                              aug_transform=self.aug_train_transform,
                              img_transform=self.img_transform,
                              configer=self.configer),
                batch_size=self.configer.get('data', 'train_batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True)

            return trainloader

        elif self.configer.get('method') == 'faster_rcnn':
            trainloader = data.DataLoader(
                FRDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                             aug_transform=self.aug_train_transform,
                             img_transform=self.img_transform,
                             configer=self.configer),
                batch_size=self.configer.get('data', 'train_batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True)

            return trainloader

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None

    def get_valloader(self):
        if self.configer.get('method') == 'single_shot_detector':
            valloader = data.DataLoader(
                SSDDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'val'),
                              aug_transform=self.aug_val_transform,
                              img_transform=self.img_transform,
                              configer=self.configer),
                batch_size=self.configer.get('data', 'val_batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True)

            return valloader

        elif self.configer.get('method') == 'faster_rcnn':
            valloader = data.DataLoader(
                FRDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'val'),
                             aug_transform=self.aug_val_transform,
                             img_transform=self.img_transform,
                             configer=self.configer),
                batch_size=self.configer.get('data', 'val_batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True)

            return valloader

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None