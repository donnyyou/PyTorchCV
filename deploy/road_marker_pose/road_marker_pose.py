#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Deploy code for Road Marker Pose.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import math
import json
import argparse
import numpy as np
from PIL import Image

from methods.det.single_shot_detector_deploy import SingleShotDetectorDeploy
from methods.pose.open_pose_deploy import OpenPoseDeploy
from deploy.road_marker_pose.src.completor import Completor


NAME_COUNT_DICT = {
    'arrow_straight': 7,'arrow_left': 9, 'arrow_right': 9, 'arrow_straight_left': 14,
    'arrow_straight_right': 14, 'arrow_straight_left_right': 21, 'arrow_left_right': 15,
    'arrow_uturn': 13, 'arrow_uturn_left': 19, 'arrow_uturn_straight': 17, 'arrow_merge_left': 9,
    'arrow_merge_right': 9
}

GROUP_DICT = {
    'arrow_straight': [
        [0, 1, 2, 5, 6, 3, 4],
    ],
    'arrow_left': [
        [0, 1, 2, 7, 8, 4, 5]
    ],
    'arrow_right': [
        [0, 1, 2, 7, 8, 4, 5],
    ],
    'arrow_straight_left': [
        [6, 7, 8, 4, 5, 10, 11], [0, 1, 2, 12, 13, 10, 11],
    ],
    'arrow_straight_right': [
        [0, 1, 2, 12, 13, 3, 4], [8, 9, 10, 6, 7, 3, 4],
    ],
    'arrow_straight_left_right': [
        [6, 7, 8, 4, 5, 10, 11], [0, 1, 2, 19, 20, 10, 11], [15, 16, 17, 13, 14, 10, 11],
    ],
    'arrow_left_right': [
        [3, 4, 5, 1, 2, 7, 8], [12, 13, 14, 10, 11, 7, 8],
    ],
    'arrow_uturn': [
        [0, 1, 2, 11, 12, 6, 7],
    ],
    'arrow_uturn_left': [
        [0, 1, 2, 17, 18, 14, 15], [9, 10, 11, 7, 8, 14, 15],
    ],
    'arrow_uturn_straight': [
        [8, 9, 10, 6, 7, 13, 14], [0, 1, 2, 15, 16, 13, 14],
    ],
    'arrow_merge_left': [
        [0, 1, 2, 7, 8, 4, 5],
    ],
    'arrow_merge_right':[
        [0, 1, 2, 7, 8, 4, 5],
    ]
}

PARAM_DICT = {
    'ssd_lane_json': '/home/donny/Projects/PytorchCV/hypes/det/lane/ssd_lane_det.json',
    'op_lane_json': '/home/donny/Projects/PytorchCV/hypes/pose/lane/op_lane_pose.json',
    'op_pole_json': '/home/donny/Projects/PytorchCV/hypes/pose/pole/op_pole_pose.json',
    'ssd_lane_model': '/home/donny/Projects/PytorchCV/checkpoints/det/lane/model_epoch199.pkl',
    'op_lane_model': '/home/donny/Projects/PytorchCV/checkpoints/pose/lane/mobile_pose_57000.pth',
    'op_pole_model': '/home/donny/Projects/PytorchCV/checkpoints/pose/pole/op_pole_pose_33000.pth'
}


class RoadMarkerPose(object):
    def __init__(self):
        self.road_marker_detector = SingleShotDetectorDeploy(PARAM_DICT['ssd_lane_json'],
                                                             PARAM_DICT['ssd_lane_model'])

        self.road_marker_poser = OpenPoseDeploy(PARAM_DICT['op_lane_json'],
                                                PARAM_DICT['op_lane_model'])

        self.road_pole_poser = OpenPoseDeploy(PARAM_DICT['op_pole_json'], PARAM_DICT['op_pole_model'])

        self.road_marker_detector.init_model([0])
        self.road_marker_poser.init_model([0])
        self.road_pole_poser.init_model([0])
        self.completor = Completor()

    def detect_pose(self, img_dir, json_save_dir):
        for img_file in os.listdir(img_dir):
            shotname, extension = os.path.splitext(img_file)
            json_file = os.path.join(json_save_dir, '{}.json'.format(shotname))
            image_path = os.path.join(img_dir, img_file)
            img_rgb = np.array(Image.open(image_path).convert('RGB'))

            json_dict = dict()
            height, width, _ = img_rgb.shape
            json_dict['image_height'] = height
            json_dict['image_width'] = width
            object_list = list()

            # img_result = img_bgr.copy()
            pose_input_size = self.road_marker_poser.configer.get('data', 'input_size')
            img_result, labels, scores, bboxes, has_obj = self.road_marker_detector.inference(img_rgb)
            image1, out_list = self.road_pole_poser.inference(img_rgb)
            img_result = self._draw_poses(out_list, img_result, self.road_pole_poser)
            # cv2.imshow('main', img_result)
            # cv2.waitKey()

            if has_obj:
                for index in range(len(bboxes)):
                    label_str = self.road_marker_detector.configer.get('details', 'name_seq')[labels[index]-1]
                    if 'arrow' not in label_str:
                        continue

                    crop_img_rgb, relate_bbox = self._crop_bbox(img_rgb, bboxes[index])
                    img_canvas, out_list = self.road_marker_poser.inference(crop_img_rgb)
                    # img_result[relate_bbox[1]:relate_bbox[3], relate_bbox[0]:relate_bbox[2], :] = img_canvas
                    for i in range(len(out_list)):
                        for j in range(self.road_marker_poser.configer.get('data', 'num_keypoints')):
                            if out_list[i]['keypoints'][j*2] == -1:
                                continue

                            out_list[i]['keypoints'][j*2] = out_list[i]['keypoints'][j*2] / pose_input_size[0] * relate_bbox[2] + relate_bbox[0]
                            out_list[i]['keypoints'][j*2+1] = out_list[i]['keypoints'][j*2+1] / pose_input_size[1] * relate_bbox[3] + relate_bbox[1]

                    img_result, object_dict = self._fit_pose(img_result, out_list, label_str)

                    if 'label' in object_dict:
                        object_list.append(object_dict)

                    img_result = self._draw_poses(out_list, img_result, self.road_marker_poser)
                    # cv2.imshow('main', img_result)
                    # cv2.waitKey()

            json_dict['objects'] = object_list

            with open(json_file, 'w') as json_stream:
                json_stream.write(json.dumps(json_dict))

            cv2.namedWindow("main", cv2.WINDOW_NORMAL)
            cv2.imshow('main', img_result)
            cv2.waitKey()

    def _fit_pose(self, img_result, out_list, label_str):

        root_array = list()
        root_score = 0.0
        arrow_list = list()
        object_dict = dict()
        print (label_str)

        for pose_dict in out_list:
            if pose_dict['num_keypoints'] == 2 and pose_dict['keypoints'][-1] != -1:
                if len(root_array) == 0 or root_score < pose_dict['score']:
                    root_score = pose_dict['score']
                    root_array = [[pose_dict['keypoints'][-2 * 2], pose_dict['keypoints'][-2 * 2 + 1]],
                                  [pose_dict['keypoints'][-1 * 2], pose_dict['keypoints'][-1 * 2 + 1]],]

            else:
                valid_arrow = True
                for i in range(5):
                    if pose_dict['keypoints'][2*i] == -1:
                        valid_arrow = False

                if valid_arrow:
                    arrow_array = list()
                    for i in range(5):
                        arrow_array.append([pose_dict['keypoints'][i*2], pose_dict['keypoints'][i*2+1]])

                    arrow_list.append(arrow_array)

        if len(GROUP_DICT[label_str]) == len(arrow_list) > 0 and len(root_array) > 0:
            select_list = list()
            used_list = list()
            pose_array = list()
            for i in range(len(GROUP_DICT[label_str])):
                max_score = -1
                max_index = -1

                for j in range(len(arrow_list)):
                    if max_score == -1:
                        max_score = sum(arrow_list[0][:][0])

                    if sum(arrow_list[j][:][0]) <= max_score and j not in select_list:
                        max_score = sum(arrow_list[j][:][0])
                        max_index = j

                select_list.append(max_index)

                for index in range(len(GROUP_DICT[label_str][i])-2):
                    if GROUP_DICT[label_str][i][index] not in used_list:
                        used_list.append(GROUP_DICT[label_str][i][index])
                        pose_array.append(arrow_list[max_index][index])

            print (used_list)
            used_list.append(GROUP_DICT[label_str][0][-2])
            used_list.append(GROUP_DICT[label_str][0][-1])

            object_dict['label'] = label_str
            object_dict['occlusion_list'] = list()
            for i in range(NAME_COUNT_DICT[label_str]):
                object_dict['occlusion_list'].append(False)

            object_dict['polygon'] = (np.ones((NAME_COUNT_DICT[label_str], 2)) * -1).tolist()

            for id in range(len(used_list)):
                object_dict['occlusion_list'][used_list[id]] = True
                object_dict['polygon'][used_list[id]][0] = np.array(pose_array+root_array)[id][0]
                object_dict['polygon'][used_list[id]][1] = np.array(pose_array+root_array)[id][1]

            proj_points = self.completor(label_str, np.array(pose_array+root_array), used_list)

            proj_points = np.int32(proj_points[0:2, :].transpose())
            cv2.polylines(img_result, [proj_points], isClosed=True, color=(255, 0, 0), thickness=3)
            # cv2.imshow('image', img_result)
            # cv2.waitKey(0)

        return img_result, object_dict

    def _draw_poses(self, pose_list, img_canvas, poser):

        for i in range(poser.configer.get('data', 'num_keypoints')):
            for pose_set in pose_list:
                if -1 == pose_set['keypoints'][i*2]:
                    continue

                cv2.circle(img_canvas, (int(pose_set['keypoints'][i*2]), int(pose_set['keypoints'][i*2+1])),
                           poser.configer.get('vis', 'circle_radius'),
                           poser.configer.get('details', 'color_list')[i], thickness=-1)

        for i in range(len(poser.configer.get('details', 'limb_seq'))):
            for n in range(len(pose_list)):
                limb_id = np.array(poser.configer.get('details', 'limb_seq')[i]) - 1
                pose_set = pose_list[n]

                if -1 == pose_set['keypoints'][limb_id[0]*2] or -1 == pose_set['keypoints'][limb_id[1]*2]:
                    continue

                Y = [pose_set['keypoints'][limb_id[0]*2+1], pose_set['keypoints'][limb_id[1]*2+1]]
                X = [pose_set['keypoints'][limb_id[0]*2], pose_set['keypoints'][limb_id[1]*2]]
                mX = np.mean(X)
                mY = np.mean(Y)
                cur_canvas = img_canvas.copy()
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                polygon = cv2.ellipse2Poly((int(mX), int(mY)),
                                           (int(length / 2),
                                            poser.configer.get('vis', 'stick_width')),
                                           int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon,
                                   poser.configer.get('details', 'color_list')[i])
                img_canvas = cv2.addWeighted(img_canvas, 0.4, cur_canvas, 0.6, 0)

        return img_canvas

    def _crop_bbox(self, img_rgb, bbox):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        left_x = int(max(0, bbox[0] - width // 8))
        right_x = int(min(img_rgb.shape[1] - 1, bbox[2] + width // 8))
        top_y = int(max(0, bbox[1] - height // 8))
        bottom_y = int(min(img_rgb.shape[0] - 1, bbox[3] + height // 8))
        crop_img_rgb = img_rgb[top_y:bottom_y, left_x:right_x, :]
        crop_img_rgb = cv2.resize(crop_img_rgb, tuple(self.road_marker_poser.configer.get('data', 'input_size')))
        # cv2.imshow('main', crop_img_rgb)
        # cv2.waitKey()

        bbox = [left_x, top_y, right_x-left_x, bottom_y-top_y]

        return crop_img_rgb, bbox

    def vis_result(self, img_rgb, bboxes, labels, scores, poses):
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='/data/DataSet/road_marker_5000/image', type=str,
                        dest='image_dir', help='The image directory.')
    parser.add_argument('--json_save_dir', default='/home/donny/temp', type=str,
                        dest='json_save_dir', help='The directory to save json files.')

    args_parser = parser.parse_args()
    if not os.path.exists(args_parser.json_save_dir):
        os.makedirs(args_parser.json_save_dir)

    road_marker_pose = RoadMarkerPose()
    road_marker_pose.detect_pose(args_parser.image_dir, args_parser.json_save_dir)