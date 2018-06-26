#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Test class for convolutional pose machine.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter

from datasets.pose_data_loader import PoseDataLoader
from datasets.tools.pose_transforms import PadImage
from datasets.tools.transforms import Normalize, ToTensor, DeNormalize
from methods.tools.module_utilizer import ModuleUtilizer
from models.pose_model_manager import PoseModelManager
from utils.helpers.image_helper import ImageHelper
from utils.helpers.file_helper import FileHelper
from utils.tools.logger import Logger as Log
from vis.visualizer.pose_visualizer import PoseVisualizer


class ConvPoseMachineTest(object):
    def __init__(self, configer):
        self.configer = configer
        self.pose_vis = PoseVisualizer(configer)
        self.pose_model_manager = PoseModelManager(configer)
        self.pose_data_loader = PoseDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.pose_net = None

    def init_model(self):
        self.pose_net = self.pose_model_manager.multi_pose_detector()
        self.pose_net = self.module_utilizer.load_net(self.pose_net)
        self.pose_net.eval()

    def __test_img(self, image_path, save_path):
        image_raw = ImageHelper.cv2_open_bgr(image_path)
        inputs = ImageHelper.bgr2rgb(image_raw)
        heatmap_avg = self.__get_heatmap(inputs)
        all_peaks = self.__extract_heatmap_info(heatmap_avg)
        image_save = self.__draw_key_point(all_peaks, image_raw)
        cv2.imwrite(save_path, image_save)

    def __get_heatmap(self, img_raw):
        multiplier = [scale * self.configer.get('data', 'input_size')[0] / img_raw.shape[1]
                      for scale in self.configer.get('data', 'scale_search')]

        heatmap_avg = np.zeros((img_raw.shape[0], img_raw.shape[1], self.configer.get('network', 'heatmap_out')))

        for i, scale in enumerate(multiplier):
            img_test = cv2.resize(img_raw, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img_test_pad, pad = PadImage(self.configer.get('network', 'stride'))(img_test)
            img_test_pad = ToTensor()(img_test_pad)
            img_test_pad = Normalize(mean=self.configer.get('trans_params', 'mean'),
                                     std=self.configer.get('trans_params', 'std'))(img_test_pad)
            with torch.no_grad():
                img_test_pad = img_test_pad.unsqueeze(0).to(self.device)
                heatmap_out_list = self.pose_net(img_test_pad)

            heatmap_out = heatmap_out_list[-1]

            # extract outputs, resize, and remove padding
            heatmap = heatmap_out.data.squeeze().cpu().numpy().transpose(1, 2, 0)
            heatmap = cv2.resize(heatmap, (0, 0), fx=self.configer.get('network', 'stride'),
                                 fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:img_test_pad.size(2) - pad[3], :img_test_pad.size(3) - pad[2], :]

            heatmap = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        return heatmap_avg

    def __extract_heatmap_info(self, heatmap_avg):
        all_peaks = []

        for part in range(self.configer.get('network', 'heatmap_out') - 1):
            map_ori = heatmap_avg[:, :, part]
            map_gau = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map_gau.shape)
            map_left[1:, :] = map_gau[:-1, :]
            map_right = np.zeros(map_gau.shape)
            map_right[:-1, :] = map_gau[1:, :]
            map_up = np.zeros(map_gau.shape)
            map_up[:, 1:] = map_gau[:, :-1]
            map_down = np.zeros(map_gau.shape)
            map_down[:, :-1] = map_gau[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map_gau >= map_left, map_gau >= map_right, map_gau >= map_up,
                 map_gau >= map_down, map_gau > self.configer.get('vis', 'part_threshold')))

            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
            peaks = list(peaks)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

            all_peaks.append(peaks_with_score)

        return all_peaks

    def __draw_key_point(self, all_peaks, img_raw):
        img_canvas = img_raw.copy()  # B,G,R order

        for i in range(self.configer.get('network', 'heatmap_out') - 1):
            for j in range(len(all_peaks[i])):
                cv2.circle(img_canvas, all_peaks[i][j][0:2], self.configer.get('vis', 'stick_width'),
                           self.configer.get('details', 'color_list')[i], thickness=-1)

        return img_canvas

    def debug(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/pose', self.configer.get('dataset'), 'debug')

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        val_data_loader = self.pose_data_loader.get_valloader()

        for i, (inputs, heatmap) in enumerate(val_data_loader):
            for j in range(inputs.size(0)):
                ori_img = DeNormalize(mean=self.configer.get('trans_params', 'mean'),
                                      std=self.configer.get('trans_params', 'std'))(inputs[j])
                image_raw = ori_img.numpy().transpose(1, 2, 0)
                image_raw = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)
                heatmap_avg = heatmap[j].numpy().transpose(1, 2, 0)
                heatmap_avg = cv2.resize(heatmap_avg, (0, 0), fx=self.configer.get('network', 'stride'),
                                     fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
                all_peaks = self.__extract_heatmap_info(heatmap_avg)
                image_save = self.__draw_key_point(all_peaks, image_raw)
                cv2.imwrite(os.path.join(base_dir, '{}_{}_result.jpg'.format(i, j)), image_save)

    def test(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/pose', self.configer.get('dataset'))

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
            base_dir = os.path.join(base_dir, 'test_dir',  test_dir.rstrip('/').split('/')[-1])
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            for filename in FileHelper.list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                save_path = os.path.join(base_dir, filename)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                self.__test_img(image_path, save_path)

    def __create_coco_submission(self, test_dir=None, base_dir=None):
        pass

    def create_submission(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/pose', self.configer.get('dataset'), 'submission')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        test_dir = self.configer.get('test_dir')
        if self.configer.get('dataset') == 'coco':
            self.__create_coco_submission(test_dir)
        else:
            Log.error('Dataset: {} is not valid.'.format(self.configer.get('dataset')))
            exit(1)

