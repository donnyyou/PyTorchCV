#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Deploy code for Road Marker Pose.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

from methods.pose.open_pose_deploy import OpenPoseDeploy
from utils.helpers.image_helper import ImageHelper
from vis.parser.pose_parser import PoseParser

PARAM_DICT = {
    'op_pole_model': '/home/donny/Projects/PytorchCV/checkpoints/pose/pole/op_pole_pose_iters5000.pth'
}

CAT_LIST = ['lamp_pole', 'traffic_light_pole', 'traffic_sign_pole_small', 'traffic_sign_pole_large',
            'camera_pole', 'digital_display_pole', 'gantry_pole', 'telegraph_pole', 'signal_pole',
            'billboard_pole', 'other_type_pole']


class RoadPolePose(object):
    def __init__(self):
        self.road_pole_poser = OpenPoseDeploy(PARAM_DICT['op_pole_model'], gpu_list=[0])
        self._adjust_deploy_params()
        self.pose_parser = PoseParser(self.road_pole_poser.configer)

    def _adjust_deploy_params(self):
        self.road_pole_poser.configer.update_value(('vis', 'part_threshold'), 0.1)
        self.road_pole_poser.configer.update_value(('vis', 'limb_threshold'), 0.1)

    def detect_pose(self, img_dir, json_save_dir):
        for img_file in os.listdir(img_dir):
            shotname, extension = os.path.splitext(img_file)
            json_file = os.path.join(json_save_dir, '{}.json'.format(shotname))
            image_path = os.path.join(img_dir, img_file)
            ori_img_rgb = ImageHelper.img2np(ImageHelper.pil_read_image(image_path))
            cur_img_rgb = ImageHelper.resize(ori_img_rgb,
                                             self.road_pole_poser.configer.get('data', 'input_size'),
                                             interpolation=Image.CUBIC)
            json_dict = self.road_pole_poser.inference(cur_img_rgb)
            height, width, _ = ori_img_rgb.shape
            json_dict['image_height'] = height
            json_dict['image_width'] = width

            for i in range(len(json_dict['objects'])):
                for index in range(len(json_dict['objects'][i]['keypoints'])):
                    if json_dict['objects'][i]['keypoints'][index][2] == -1:
                        continue

                    json_dict['objects'][i]['keypoints'][index][0] *= (width / cur_img_rgb.shape[1])
                    json_dict['objects'][i]['keypoints'][index][1] *= (height / cur_img_rgb.shape[0])

                out_keypoints = json_dict['objects'][i]['keypoints']
                json_dict['objects'][i]['pole_segment'] = np.zeros((2, 2)).tolist()
                for j in range(self.road_pole_poser.configer.get('data', 'num_keypoints')):
                    if out_keypoints[j][2] == -1:
                        continue

                    else:
                        json_dict['objects'][i]['pole_segment'][0][0] = out_keypoints[j][0]
                        json_dict['objects'][i]['pole_segment'][0][1] = out_keypoints[j][1]
                        json_dict['objects'][i]['pole_segment'][1][0] = out_keypoints[j+1][0]
                        json_dict['objects'][i]['pole_segment'][1][1] = out_keypoints[j+1][1]
                        json_dict['objects'][i]['label'] = CAT_LIST[j // 2]
                        json_dict['objects'][i]['id'] = i
                        break

            with open(json_file, 'w') as json_stream:
                json_stream.write(json.dumps(json_dict))

            image_canvas = self.pose_parser.draw_points(ImageHelper.rgb2bgr(ori_img_rgb), json_dict)
            image_result = self.pose_parser.link_points(image_canvas, json_dict)

            cv2.imshow('main', image_result)
            cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default=None, type=str,
                        dest='image_dir', help='The image directory.')
    parser.add_argument('--json_save_dir', default=None, type=str,
                        dest='json_save_dir', help='The directory to save json files.')

    args_parser = parser.parse_args()
    if not os.path.exists(args_parser.json_save_dir):
        os.makedirs(args_parser.json_save_dir)

    road_pole_pose = RoadPolePose()
    road_pole_pose.detect_pose(args_parser.image_dir, args_parser.json_save_dir)