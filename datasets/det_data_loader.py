#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch.utils import data

import datasets.tools.pil_aug_transforms as pil_aug_trans
import datasets.tools.cv2_aug_transforms as cv2_aug_trans
import datasets.tools.transforms as trans
from datasets.det.ssd_data_loader import SSDDataLoader
from datasets.det.fr_data_loader import FRDataLoader
from datasets.det.yolo_data_loader import YOLODataLoader
from utils.tools.logger import Logger as Log


class DetDataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_train_transform = pil_aug_trans.PILAugCompose(self.configer, split='train')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='train')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_val_transform = pil_aug_trans.PILAugCompose(self.configer, split='val')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_val_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='val')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(div_value=self.configer.get('trans_params', 'normalize')['div_value'],
                            mean=self.configer.get('trans_params', 'normalize')['mean'],
                            std=self.configer.get('trans_params', 'normalize')['std']), ])

    def get_trainloader(self):
        if self.configer.get('method') == 'single_shot_detector':
            trainloader = data.DataLoader(
                SSDDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                              aug_transform=self.aug_train_transform,
                              img_transform=self.img_transform,
                              configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._det_collate, pin_memory=True)

            return trainloader

        elif self.configer.get('method') == 'faster_rcnn':
            trainloader = data.DataLoader(
                FRDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                             aug_transform=self.aug_train_transform,
                             img_transform=self.img_transform,
                             configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._det_collate, pin_memory=True)

            return trainloader

        elif self.configer.get('method') == 'yolov3':
            trainloader = data.DataLoader(
                YOLODataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'train'),
                               aug_transform=self.aug_train_transform,
                               img_transform=self.img_transform,
                               configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), shuffle=True,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._det_collate, pin_memory=True)

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
                batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._det_collate, pin_memory=True)

            return valloader

        elif self.configer.get('method') == 'faster_rcnn':
            valloader = data.DataLoader(
                FRDataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'val'),
                             aug_transform=self.aug_val_transform,
                             img_transform=self.img_transform,
                             configer=self.configer),
                batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._det_collate, pin_memory=True)

            return valloader

        elif self.configer.get('method') == 'yolov3':
            valloader = data.DataLoader(
                YOLODataLoader(root_dir=os.path.join(self.configer.get('data', 'data_dir'), 'val'),
                               aug_transform=self.aug_val_transform,
                               img_transform=self.img_transform,
                               configer=self.configer),
                batch_size=self.configer.get('val', 'batch_size'), shuffle=False,
                num_workers=self.configer.get('data', 'workers'), collate_fn=self._det_collate, pin_memory=True)

            return valloader

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None

    @staticmethod
    def _det_collate(batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).
        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations
        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on 0 dim
        """
        out_list= []
        for i in range(len(batch[0])):
            out_list.append([])

        for sample in batch:
            for i in range(len(sample)):
                out_list[i].append(sample[i])

        return out_list
