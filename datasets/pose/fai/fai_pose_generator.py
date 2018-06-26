#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import json
import shutil
import argparse
import numpy as np
import csv
from PIL import Image


JOSN_DIR = 'json'
MASK_DIR = 'mask'
IMAGE_DIR = 'image'

Cat_List = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']


class FaiPoseGenerator(object):

    def __init__(self, args, json_dir=JOSN_DIR, mask_dir=MASK_DIR, image_dir=IMAGE_DIR):
        self.args = args
        self.json_dir = os.path.join(self.args.root_dir, json_dir)
        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir)

        self.image_dir = os.path.join(self.args.root_dir, image_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def generate_label(self):
        with open(self.args.anno_file, 'rb') as file_stream:
            for line in csv.reader(file_stream):
                if line[0] == 'image_id':
                    continue

                img_path = os.path.join(self.args.data_dir, line[0])
                filename = line[0].rstrip().split('/')[-1]
                category = line[0].rstrip().split('/')[1]
                save_json_dir = os.path.join(self.json_dir, category)

                if not os.path.exists(save_json_dir):
                    os.makedirs(save_json_dir)

                save_image_dir = os.path.join(self.image_dir, category)

                if not os.path.exists(save_image_dir):
                    os.makedirs(save_image_dir)

                json_dict = dict()
                object_list = list()
                object_dict = dict()
                object_dict['keypoints'] = np.zeros((24, 3)).tolist()
                for part in range(24):
                    items = line[part+2].strip().split('_')
                    object_dict['keypoints'][part][0] = int(items[0])
                    object_dict['keypoints'][part][1] = int(items[1])
                    object_dict['keypoints'][part][2] = int(items[2])

                object_list.append(object_dict)
                json_dict['objects'] = object_list
                json_dict['category'] = category

                fw = open(os.path.join(save_json_dir, '{}.json'.format(filename.split('.')[0])), 'w')
                fw.write(json.dumps(json_dict))
                fw.close()
                shutil.copyfile(img_path, os.path.join(save_image_dir, filename))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default=None, type=str,
                        dest='root_dir', help='The directory to save the ground truth.')
    parser.add_argument('--anno_file', default=None, type=str,
                        dest='anno_file', help='The annotations file of coco keypoints.')
    parser.add_argument('--data_dir', default=None, type=str,
                        dest='data_dir', help='The data dir corresponding to coco anno file.')

    args = parser.parse_args()

    fai_pose_generator = FaiPoseGenerator(args)
    fai_pose_generator.generate_label()