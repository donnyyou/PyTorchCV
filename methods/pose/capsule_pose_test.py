#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Pose Estimator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from datasets.pose_data_loader import PoseDataLoader
from datasets.tools.pose_transforms import PadImage
from datasets.tools.transforms import Normalize, ToTensor, DeNormalize
from methods.tools.module_utilizer import ModuleUtilizer
from models.pose_model_manager import PoseModelManager
from utils.helpers.image_helper import ImageHelper
from utils.helpers.file_helper import FileHelper
from utils.tools.logger import Logger as Log
from vis.parser.pose_parser import PoseParser
from vis.visualizer.pose_visualizer import PoseVisualizer


class CapsulePoseTest(object):
    def __init__(self, configer):
        self.configer = configer

        self.pose_visualizer = PoseVisualizer(configer)
        self.pose_parser = PoseParser(configer)
        self.pose_model_manager = PoseModelManager(configer)
        self.pose_data_loader = PoseDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.pose_net = None

    def init_model(self):
        self.pose_net = self.pose_model_manager.multi_pose_detector()
        self.pose_net = self.module_utilizer.load_net(self.pose_net)
        self.pose_net.eval()

    def __test_img(self, image_path, json_path, raw_path, vis_path):
        Log.info('Image Path: {}'.format(image_path))
        ori_img_rgb = ImageHelper.img2np(ImageHelper.pil_open_rgb(image_path))
        cur_img_rgb = ImageHelper.resize(ori_img_rgb,
                                         self.configer.get('data', 'input_size'),
                                         interpolation=Image.CUBIC)

        ori_img_bgr = ImageHelper.bgr2rgb(ori_img_rgb)
        paf_avg, heatmap_avg, partmap_avg = self.__get_paf_and_heatmap(cur_img_rgb)
        all_peaks = self.__extract_heatmap_info(heatmap_avg)
        special_k, connection_all = self.__extract_paf_info(cur_img_rgb, paf_avg, partmap_avg, all_peaks)
        subset, candidate = self.__get_subsets(connection_all, special_k, all_peaks)
        json_dict = self.__get_info_tree(cur_img_rgb, subset, candidate)
        for i in range(len(json_dict['objects'])):
            for index in range(len(json_dict['objects'][i]['keypoints'])):
                if json_dict['objects'][i]['keypoints'][index][2] == -1:
                    continue

                json_dict['objects'][i]['keypoints'][index][0] *= (ori_img_rgb.shape[1] / cur_img_rgb.shape[1])
                json_dict['objects'][i]['keypoints'][index][1] *= (ori_img_rgb.shape[0] / cur_img_rgb.shape[0])

        image_canvas = self.pose_parser.draw_points(ori_img_bgr.copy(), json_dict)
        image_canvas = self.pose_parser.link_points(image_canvas, json_dict)

        cv2.imwrite(vis_path, image_canvas)
        cv2.imwrite(raw_path, ori_img_bgr)
        Log.info('Json Save Path: {}'.format(json_path))
        with open(json_path, 'w') as save_stream:
            save_stream.write(json.dumps(json_dict))

    def __get_info_tree(self, image_raw, subset, candidate):
        json_dict = dict()
        height, width, _ = image_raw.shape
        json_dict['image_height'] = height
        json_dict['image_width'] = width
        object_list = list()
        for n in range(len(subset)):
            if subset[n][-1] <= 1:
                continue

            object_dict = dict()
            object_dict['keypoints'] = np.zeros((self.configer.get('data', 'num_keypoints'), 3)).tolist()
            for j in range(self.configer.get('data', 'num_keypoints')):
                index = subset[n][j]
                if index == -1:
                    object_dict['keypoints'][j][0] = -1
                    object_dict['keypoints'][j][1] = -1
                    object_dict['keypoints'][j][2] = -1

                else:
                    object_dict['keypoints'][j][0] = candidate[index.astype(int)][0]
                    object_dict['keypoints'][j][1] = candidate[index.astype(int)][1]
                    object_dict['keypoints'][j][2] = 1

            object_dict['score'] = subset[n][-2]
            object_list.append(object_dict)

        json_dict['objects'] = object_list
        return json_dict

    def __get_paf_and_heatmap(self, img_raw):
        multiplier = [scale * self.configer.get('data', 'input_size')[0] / img_raw.shape[1]
                      for scale in self.configer.get('data', 'scale_search')]

        heatmap_avg = np.zeros((img_raw.shape[0], img_raw.shape[1], self.configer.get('data', 'num_keypoints')))
        paf_avg = np.zeros((img_raw.shape[0], img_raw.shape[1], self.configer.get('network', 'paf_out')))
        partmap_avg = np.zeros((img_raw.shape[0], img_raw.shape[1], self.configer.get('network', 'heatmap_out')))

        for i, scale in enumerate(multiplier):
            img_test = cv2.resize(img_raw, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img_test_pad, pad = PadImage(self.configer.get('network', 'stride'))(img_test)
            pad_right = pad[2]
            pad_down = pad[3]
            img_test_pad = ToTensor()(img_test_pad)
            img_test_pad = Normalize(mean=self.configer.get('trans_params', 'mean'),
                                     std=self.configer.get('trans_params', 'std'))(img_test_pad)
            with torch.no_grad():
                img_test_pad = img_test_pad.unsqueeze(0).to(self.device)
                paf_out_list, partmap_out_list = self.pose_net(img_test_pad)

            paf_out = paf_out_list[-1]
            partmap_out = partmap_out_list[-1]
            partmap = partmap_out.data.squeeze().cpu().numpy().transpose(1, 2, 0)
            paf = paf_out.data.squeeze().cpu().numpy().transpose(1, 2, 0)
            # self.pose_visualizer.vis_tensor(heatmap_out)
            heatmap = np.zeros(
                (partmap.shape[0], partmap.shape[1], self.configer.get('data', 'num_keypoints')))
            part_num = np.zeros((self.configer.get('data', 'num_keypoints'),))

            for index in range(len(self.configer.get('details', 'limb_seq'))):
                a = self.configer.get('details', 'limb_seq')[index][0] - 1
                b = self.configer.get('details', 'limb_seq')[index][1] - 1
                heatmap_a = partmap[:, :, index * 4:index * 4 + 2] ** 2
                heatmap_a = np.sqrt(np.sum(heatmap_a, axis=2).squeeze())
                heatmap[:, :, a] = (heatmap[:, :, a] * part_num[a] + heatmap_a) / (part_num[a] + 1)
                part_num[a] += 1
                heatmap_b = partmap[:, :, index * 4 + 2:index * 4 + 4] ** 2
                heatmap_b = np.sqrt(np.sum(heatmap_b, axis=2).squeeze())
                heatmap[:, :, b] = (heatmap[:, :, b] * part_num[b] + heatmap_b) / (part_num[b] + 1)
                part_num[b] += 1

            heatmap = cv2.resize(heatmap, (0, 0), fx=self.configer.get('network', 'stride'),
                                 fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:img_test_pad.size(2) - pad_down, :img_test_pad.size(3) - pad_right, :]
            heatmap = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)

            partmap = cv2.resize(partmap, (0, 0), fx=self.configer.get('network', 'stride'),
                                 fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
            partmap = partmap[:img_test_pad.size(2) - pad_down, :img_test_pad.size(3) - pad_right, :]
            partmap = cv2.resize(partmap, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)

            paf = cv2.resize(paf, (0, 0), fx=self.configer.get('network', 'stride'),
                             fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
            paf = paf[:img_test_pad.size(2) - pad_down, :img_test_pad.size(3) - pad_right, :]
            paf = cv2.resize(paf, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)

            partmap_avg = partmap_avg + partmap / len(multiplier)
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)

        return paf_avg, heatmap_avg, partmap_avg

    def __extract_heatmap_info(self, heatmap_avg):
        all_peaks = []
        peak_counter = 0

        for part in range(self.configer.get('data', 'num_keypoints')):
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
            ids = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (ids[i],) for i in range(len(ids))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        return all_peaks

    def __extract_paf_info(self, img_raw, paf_avg, partmap_avg, all_peaks):
        connection_all = []
        special_k = []
        mid_num = self.configer.get('vis', 'mid_point_num')

        for k in range(len(self.configer.get('details', 'limb_seq'))):
            score_mid = paf_avg[:, :, [k*2, k*2+1]]
            # self.pose_visualizer.vis_paf(score_mid, img_raw, name='pa{}'.format(k))
            candA = all_peaks[self.configer.get('details', 'limb_seq')[k][0] - 1]
            candB = all_peaks[self.configer.get('details', 'limb_seq')[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec_a = partmap_avg[candA[i][1], candA[i][0], k*4:k*4+2]
                        vec_b = -partmap_avg[candB[j][1], candB[j][0], k*4+2:k*4+4]
                        norm_a = math.sqrt(vec_a[0] * vec_a[0] + vec_a[1] * vec_a[1]) + 1e-9
                        vec_a = np.divide(vec_a, norm_a)
                        norm_b = math.sqrt(vec_b[0] * vec_b[0] + vec_b[1] * vec_b[1]) + 1e-9
                        vec_b = np.divide(vec_b, norm_b)

                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        sim_length = np.sum(vec_a*vec + vec_b*vec) / 2.0
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-9
                        vec = np.divide(vec, norm)

                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num))
                        startend = list(startend)

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts)
                        score_with_dist_prior += min(0.5 * img_raw.shape[0] / norm - 1, 0)

                        num_positive = len(np.nonzero(score_midpts > self.configer.get('vis', 'limb_threshold'))[0])
                        criterion1 = num_positive > int(self.configer.get('vis', 'limb_pos_ratio') * len(score_midpts))
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2 and sim_length > self.configer.get('vis', 'sim_length'):
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        return special_k, connection_all

    def __get_subsets(self, connection_all, special_k, all_peaks):
        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, self.configer.get('data', 'num_keypoints') + 2))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in self.configer.get('details', 'mini_tree'):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(self.configer.get('details', 'limb_seq')[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found:
                        row = -1 * np.ones(self.configer.get('data', 'num_keypoints') + 2)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        return subset, candidate

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
            filename = test_img.rstrip().split('/')[-1]
            json_path = os.path.join(base_dir, 'json', '{}.json'.format('.'.join(filename.split('.')[:-1])))
            raw_path = os.path.join(base_dir, 'raw', filename)
            vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
            if not os.path.exists(os.path.dirname(json_path)):
                os.makedirs(os.path.dirname(json_path))

            if not os.path.exists(os.path.dirname(raw_path)):
                os.makedirs(os.path.dirname(raw_path))

            if not os.path.exists(os.path.dirname(vis_path)):
                os.makedirs(os.path.dirname(vis_path))

            self.__test_img(test_img, json_path, raw_path, vis_path)

        else:
            base_dir = os.path.join(base_dir, 'test_dir', test_dir.rstrip('/').split('/')[-1])
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            for filename in FileHelper.list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                json_path = os.path.join(base_dir, 'json', '{}.json'.format('.'.join(filename.split('.')[:-1])))
                raw_path = os.path.join(base_dir, 'raw', filename)
                vis_path = os.path.join(base_dir, 'vis', '{}_vis.png'.format('.'.join(filename.split('.')[:-1])))
                if not os.path.exists(os.path.dirname(json_path)):
                    os.makedirs(os.path.dirname(json_path))

                if not os.path.exists(os.path.dirname(raw_path)):
                    os.makedirs(os.path.dirname(raw_path))

                if not os.path.exists(os.path.dirname(vis_path)):
                    os.makedirs(os.path.dirname(vis_path))

                self.__test_img(image_path, json_path, raw_path, vis_path)

    def debug(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/pose', self.configer.get('dataset'), 'debug')

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        val_data_loader = self.pose_data_loader.get_valloader()

        count = 0
        for i, (inputs, partmap, maskmap, vecmap) in enumerate(val_data_loader):
            for j in range(inputs.size(0)):
                count = count + 1
                if count > 2:
                    exit(1)

                Log.info(partmap.size())
                ori_img = DeNormalize(mean=self.configer.get('trans_params', 'mean'),
                                      std=self.configer.get('trans_params', 'std'))(inputs[j])
                ori_img = ori_img.numpy().transpose(1, 2, 0).astype(np.uint8)
                image_bgr = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
                partmap_avg = partmap[j].numpy().transpose(1, 2, 0)

                heatmap_avg = np.zeros(
                    (partmap_avg.shape[0], partmap_avg.shape[1], self.configer.get('data', 'num_keypoints')))
                part_num = np.zeros((self.configer.get('data', 'num_keypoints'),))

                for index in range(len(self.configer.get('details', 'limb_seq'))):
                    a = self.configer.get('details', 'limb_seq')[index][0] - 1
                    b = self.configer.get('details', 'limb_seq')[index][1] - 1
                    heatmap_a = partmap_avg[:, :, index * 4:index * 4 + 2] ** 2
                    heatmap_a = np.sqrt(np.sum(heatmap_a, axis=2).squeeze())
                    heatmap_avg[:, :, a] = (heatmap_avg[:, :, a] * part_num[a] + heatmap_a) / (part_num[a] + 1)
                    part_num[a] += 1
                    heatmap_b = partmap_avg[:, :, index * 4 + 2:index * 4 + 4] ** 2
                    heatmap_b = np.sqrt(np.sum(heatmap_b, axis=2).squeeze())
                    heatmap_avg[:, :, b] = (heatmap_avg[:, :, b] * part_num[b] + heatmap_b) / (part_num[b] + 1)
                    part_num[b] += 1

                partmap_avg = cv2.resize(partmap_avg, (0, 0), fx=self.configer.get('network', 'stride'),
                                         fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
                heatmap_avg = cv2.resize(heatmap_avg, (0, 0), fx=self.configer.get('network', 'stride'),
                                         fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
                paf_avg = vecmap[j].numpy().transpose(1, 2, 0)
                paf_avg = cv2.resize(paf_avg, (0, 0), fx=self.configer.get('network', 'stride'),
                                     fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)

                self.pose_visualizer.vis_peaks(heatmap_avg, image_bgr)
                self.pose_visualizer.vis_paf(paf_avg, image_bgr)
                all_peaks = self.__extract_heatmap_info(heatmap_avg)
                special_k, connection_all = self.__extract_paf_info(image_bgr, paf_avg, partmap_avg, all_peaks)
                subset, candidate = self.__get_subsets(connection_all, special_k, all_peaks)
                json_dict = self.__get_info_tree(image_bgr, subset, candidate)
                image_canvas = self.pose_parser.draw_points(image_bgr, json_dict)
                image_canvas = self.pose_parser.link_points(image_canvas, json_dict)
                cv2.imwrite(os.path.join(base_dir, '{}_{}_vis.png'.format(i, j)), image_canvas)
                cv2.imshow('main', image_canvas)
                cv2.waitKey()
