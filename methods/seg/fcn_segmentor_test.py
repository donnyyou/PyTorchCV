#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.seg_data_loader import SegDataLoader
from datasets.tools.seg_transforms import Scale
from datasets.tools.transforms import ToTensor, Normalize, DeNormalize
from methods.tools.module_utilizer import ModuleUtilizer
from models.seg_model_manager import SegModelManager
from utils.helpers.image_helper import ImageHelper
from utils.helpers.file_helper import FileHelper
from utils.tools.logger import Logger as Log
from vis.parser.seg_parser import SegParser
from vis.visualizer.seg_visualizer import SegVisualizer


class FCNSegmentorTest(object):
    def __init__(self, configer):
        self.configer = configer

        self.seg_visualizer = SegVisualizer(configer)
        self.seg_parser = SegParser(configer)
        self.seg_model_manager = SegModelManager(configer)
        self.seg_data_loader = SegDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.seg_net = None

    def init_model(self):
        self.seg_net = self.seg_model_manager.semantic_segmentor()
        self.seg_net = self.module_utilizer.load_net(self.seg_net)
        self.seg_net.eval()

    def __test_img(self, image_path, save_path):
        image = ImageHelper.pil_open_rgb(image_path)
        ori_width, ori_height = image.size
        image = Scale(size=self.configer.get('data', 'input_size'))(image)
        image = ToTensor()(image)
        image = Normalize(mean=self.configer.get('trans_params', 'mean'),
                          std=self.configer.get('trans_params', 'std'))(image)
        with torch.no_grad():
            inputs = image.unsqueeze(0).to(self.device)
            results = self.seg_net.forward(inputs)

            label_map = results.data.cpu().numpy().argmax(axis=1)[0].squeeze()

            label_img = np.array(label_map, dtype=np.uint8)
            if not self.configer.is_empty('details', 'label_list'):
                label_img = self.__relabel(label_img)

            label_img = Image.fromarray(label_img, 'P')
            label_img = label_img.resize((ori_width, ori_height), Image.NEAREST)
            label_img.save(save_path)

    def __relabel(self, label_map):
        height, width = label_map.shape
        label_dst = np.zeros((height, width), dtype=np.uint8)
        for i in range(self.configer.get('data', 'num_classes')):
            label_dst[label_map == i] = self.configer.get('details', 'label_list')[i]

        label_dst = np.array(label_dst, dtype=np.uint8)

        return label_dst

    def test(self):
        base_dir = os.path.join(self.configer.get('output_dir'),
                                'val/results/seg', self.configer.get('dataset'))

        test_img = self.configer.get('test_img')
        test_dir = self.configer.get('test_dir')
        if test_img is None and test_dir is None:
            Log.error('test_img & test_dir not exists.')
            exit(1)

        if test_img is not None and test_dir is not None:
            Log.error('Either test_img or test_dir.')
            exit(1)

        if test_img is not None:
            base_dir = os.path.join(base_dir, 'test_img')
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            filename = test_img.rstrip().split('/')[-1]
            save_path = os.path.join(base_dir, filename)
            self.__test_img(test_img, save_path)

        else:
            base_dir = os.path.join(base_dir, 'test_dir', test_dir.rstrip('/').split('/')[-1])
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            for filename in FileHelper.list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                save_path = os.path.join(base_dir, filename)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                self.__test_img(image_path, save_path)

    def debug(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/seg', self.configer.get('dataset'), 'debug')

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        val_data_loader = self.seg_data_loader.get_valloader()

        count = 0
        for i, (inputs, targets) in enumerate(val_data_loader):
            for j in range(inputs.size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                ori_img = DeNormalize(mean=self.configer.get('trans_params', 'mean'),
                                      std=self.configer.get('trans_params', 'std'))(inputs[j])
                ori_img = ori_img.numpy().transpose(1, 2, 0).astype(np.uint8)

                image_bgr = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
                label_map = targets[j].numpy()
                image_canvas = self.seg_parser.colorize(label_map, image_canvas=image_bgr)
                cv2.imwrite(os.path.join(base_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()

