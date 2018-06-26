#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Pose Estimator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter

from datasets.tools.pose_transforms import PadImage
from datasets.tools.transforms import Normalize, ToTensor
from models.pose_model_manager import PoseModelManager
from utils.helpers.image_helper import ImageHelper
from utils.tools.configer import Configer
from utils.tools.logger import Logger as Log


class OpenPoseDeploy(object):
    def __init__(self, model_path=None, gpu_list=list()):
        self.pose_net = None
        self.configer = None
        self._init_model(model_path=model_path, gpu_list=gpu_list)

    def _init_model(self, model_path, gpu_list):
        self.device = torch.device('cpu' if len(gpu_list) == 0 else 'cuda')
        model_dict = None
        if model_path is not None:
            model_dict = torch.load(model_path)
        else:
            Log.error('Model Path is not existed.')
            exit(1)

        self.configer = Configer(config_dict=model_dict['config_dict'])

        if len(gpu_list) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in gpu_list)

        self.pose_model_manager = PoseModelManager(self.configer)
        self.pose_net = self.pose_model_manager.multi_pose_detector()
        self.pose_net = nn.DataParallel(self.pose_net).to(self.device)
        self.pose_net.load_state_dict(model_dict['state_dict'])
        self.pose_net.eval()

    def inference(self, image_rgb):
        image_bgr = ImageHelper.rgb2bgr(image_rgb)
        paf_avg, heatmap_avg = self.__get_paf_and_heatmap(image_rgb)
        all_peaks = self.__extract_heatmap_info(heatmap_avg)
        special_k, connection_all = self.__extract_paf_info(image_rgb, paf_avg, all_peaks)
        subset, candidate = self.__get_subsets(connection_all, special_k, all_peaks)
        json_dict = self.__get_info_tree(image_bgr, subset, candidate)

        return json_dict

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

        heatmap_avg = np.zeros((img_raw.shape[0], img_raw.shape[1], self.configer.get('network', 'heatmap_out')))
        paf_avg = np.zeros((img_raw.shape[0], img_raw.shape[1], self.configer.get('network', 'paf_out')))

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
                paf_out_list, heatmap_out_list = self.pose_net(img_test_pad)

            paf_out = paf_out_list[-1]
            heatmap_out = heatmap_out_list[-1]

            # extract outputs, resize, and remove padding
            heatmap = heatmap_out.data.squeeze().cpu().numpy().transpose(1, 2, 0)
            heatmap = cv2.resize(heatmap,  (0, 0), fx=self.configer.get('network', 'stride'),
                                 fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:img_test_pad.size(2) - pad_down, :img_test_pad.size(3) - pad_right, :]
            heatmap = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)

            paf = paf_out.data.squeeze().cpu().numpy().transpose(1, 2, 0)
            paf = cv2.resize(paf, (0, 0), fx=self.configer.get('network', 'stride'),
                                 fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
            paf = paf[:img_test_pad.size(2) - pad_down, :img_test_pad.size(3) - pad_right, :]
            paf = cv2.resize(paf, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)

        return paf_avg, heatmap_avg

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

    def __extract_paf_info(self, img_raw, paf_avg, all_peaks):
        connection_all = []
        special_k = []
        mid_num = 10

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
                        vec = np.subtract(candB[j][:2], candA[i][:2])
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
                        criterion1 = num_positive > int(0.8 * len(score_midpts))
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
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
