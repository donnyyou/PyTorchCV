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
from pycocotools.coco import COCO
from scipy.ndimage.filters import gaussian_filter

from datasets.tools.pose_transforms import PadImage
from datasets.tools.transforms import Normalize, ToTensor
from methods.tools.module_utilizer import ModuleUtilizer
from models.pose_model_manager import PoseModelManager
from utils.helpers.image_helper import ImageHelper
from utils.helpers.file_helper import FileHelper
from utils.tools.logger import Logger as Log
from vis.visualizer.pose_visualizer import PoseVisualizer


class RPNPoseTest(object):
    def __init__(self, configer):
        self.configer = configer

        self.pose_visualizer = PoseVisualizer(configer)
        self.pose_model_manager = PoseModelManager(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.pose_net = None

    def init_model(self):
        self.pose_net = self.pose_model_manager.multi_pose_detector()
        self.pose_net = self.module_utilizer.load_net(self.pose_net)
        self.pose_net.eval()

    def __test_img(self, image_path, save_path):
        Log.info('Image Path: {}'.format(image_path))
        image_raw = ImageHelper.cv2_open_bgr(image_path)
        inputs = ImageHelper.bgr2rgb(image_raw)
        paf_avg, heatmap_avg = self.__get_paf_and_heatmap(inputs)
        all_peaks = self.__extract_heatmap_info(heatmap_avg)
        special_k, connection_all = self.__extract_paf_info(image_raw, paf_avg, all_peaks)
        subset, candidate = self.__get_subsets(connection_all, special_k, all_peaks)
        subset, img_canvas = self.__draw_key_point(subset, all_peaks, image_raw)
        img_canvas = self.__link_key_point(img_canvas, candidate, subset)
        cv2.imwrite(save_path, img_canvas)

    def __get_paf_and_heatmap(self, img_raw):
        multiplier = [scale * self.configer.get('data', 'input_size')[0] / img_raw.shape[1]
                      for scale in self.configer.get('data', 'scale_search')]

        heatmap_avg = np.zeros((img_raw.shape[0], img_raw.shape[1], self.configer.get('network', 'heatmap_out')))
        paf_avg = np.zeros((img_raw.shape[0], img_raw.shape[1], self.configer.get('network', 'paf_out')))

        for i, scale in enumerate(multiplier):
            img_test = cv2.resize(img_raw, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img_test_pad, pad = PadImage(self.configer.get('network', 'stride'))(img_test)
            img_test_pad = ToTensor()(img_test_pad)
            img_test_pad = Normalize(mean=self.configer.get('trans_params', 'mean'),
                                     std=self.configer.get('trans_params', 'std'))(img_test_pad)
            with torch.no_grad():
                img_test_pad = img_test_pad.unsqueeze(0).to(self.device)
                paf_out, heatmap_out = self.pose_net(img_test_pad)

            # extract outputs, resize, and remove padding
            heatmap = heatmap_out.data.squeeze().cpu().numpy().transpose(1, 2, 0)
            heatmap = cv2.resize(heatmap,  (0, 0), fx=self.configer.get('network', 'stride'),
                                 fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:img_test_pad.size(2) - pad[3], :img_test_pad.size(3) - pad[2], :]
            heatmap = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_CUBIC)

            paf = paf_out.data.squeeze().cpu().numpy().transpose(1, 2, 0)
            paf = cv2.resize(paf, (0, 0), fx=self.configer.get('network', 'stride'),
                                 fy=self.configer.get('network', 'stride'), interpolation=cv2.INTER_CUBIC)
            paf = paf[:img_test_pad.size(2) - pad[3], :img_test_pad.size(3) - pad[2], :]
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
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(self.configer.get('details', 'limb_seq'))):
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
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        return subset, candidate

    def __draw_key_point(self, subset, all_peaks, img_raw):
        del_ids = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                #del_ids.append(i)
                pass
        subset = np.delete(subset, del_ids, axis=0)
        img_canvas = img_raw.copy()  # B,G,R order

        for i in range(self.configer.get('data', 'num_keypoints')):
            for j in range(len(all_peaks[i])):
                cv2.circle(img_canvas, all_peaks[i][j][0:2],
                           self.configer.get('vis', 'circle_radius'),
                           self.configer.get('details', 'color_list')[i], thickness=-1)

        return subset, img_canvas

    def __link_key_point(self, img_canvas, candidate, subset):
        for i in range(self.configer.get('data', 'num_keypoints')-1):
            for n in range(len(subset)):
                index = subset[n][np.array(self.configer.get('details', 'limb_seq')[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = img_canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                           (int(length / 2),
                                            self.configer.get('vis', 'stick_width')), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, self.configer.get('details', 'color_list')[i])
                img_canvas = cv2.addWeighted(img_canvas, 0.4, cur_canvas, 0.6, 0)

        return img_canvas

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
        out_file = os.path.join(base_dir, 'person_keypoints_val2017_donny_results.json')
        out_list = list()
        coco = COCO(os.path.join(test_dir, 'person_keypoints_val2017.json'))
        for i, img_id in enumerate(list(coco.imgs.keys())):
            filename = coco.imgs[img_id]['file_name']
            image_raw = cv2.imread(os.path.join(test_dir, 'val2017', filename))
            print (i)
            paf_avg, heatmap_avg = self.__get_paf_and_heatmap(image_raw)
            all_peaks = self.__extract_heatmap_info(heatmap_avg)
            special_k, connection_all = self.__extract_paf_info(image_raw, paf_avg, all_peaks)
            subset, candidate = self.__get_subsets(connection_all, special_k, all_peaks)
            subset, img_canvas = self.__draw_key_point(subset, all_peaks, image_raw)
            img_canvas = self.__link_key_point(img_canvas, candidate, subset)
            cv2.imwrite(os.path.join(base_dir, filename), img_canvas)
            for n in range(len(subset)):
                dict_temp = dict()
                dict_temp['image_id'] = img_id
                dict_temp['category_id'] = 1
                dict_temp['score'] = subset[n][-2]
                pose_list = list()
                for i in range(self.configer.get('data', 'num_keypoints')-1):
                    index = subset[n][self.configer.get('details', 'coco_to_ours')[i]]
                    if index == -1:
                        pose_list.append(0)
                        pose_list.append(0)

                    else:
                        pose_list.append(candidate[index.astype(int)][0])
                        pose_list.append(candidate[index.astype(int)][1])

                    pose_list.append(1)

                dict_temp['keypoints'] = pose_list

                out_list.append(dict_temp)

        fw = open(out_file, 'w')
        fw.write(json.dumps(out_list))
        fw.close()

    def create_submission(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/pose', self.configer.get('dataset'), 'submission')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        test_dir = self.configer.get('test_dir')
        if self.configer.get('dataset') == 'coco':
            self.__create_coco_submission(test_dir, base_dir)
        else:
            Log.error('Dataset: {} is not valid.'.format(self.configer.get('dataset')))
            exit(1)

