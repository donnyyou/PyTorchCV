#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# VOC det data generator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import argparse
import shutil
from bs4 import BeautifulSoup


JOSN_DIR = 'json'
IMAGE_DIR = 'image'
CAT_DICT_BK = {
    'aeroplane': 0, 'bicycle':1,'bird':2,'boat':3,'bottle':4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
    'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

CAT_DICT = {
    'upper': 0, 'lower': 1, 'full':2
}


class VocDetGenerator(object):

    def __init__(self, args, json_dir=JOSN_DIR, image_dir=IMAGE_DIR):
        self.args = args
        self.train_json_dir = os.path.join(self.args.save_dir, 'train', json_dir)
        self.val_json_dir = os.path.join(self.args.save_dir, 'val', json_dir)
        if not os.path.exists(self.train_json_dir):
            os.makedirs(self.train_json_dir)

        if not os.path.exists(self.val_json_dir):
            os.makedirs(self.val_json_dir)

        self.train_image_dir = os.path.join(self.args.save_dir, 'train', image_dir)
        self.val_image_dir = os.path.join(self.args.save_dir, 'val', image_dir)
        if not os.path.exists(self.train_image_dir):
            os.makedirs(self.train_image_dir)

        if not os.path.exists(self.val_image_dir):
            os.makedirs(self.val_image_dir)

    def generate_label(self):
        file_count = 0
        for label_file in os.listdir(self.args.ori_label_dir):
            file_count += 1
            label_file_path = os.path.join(self.args.ori_label_dir, label_file)
            object_list = list()
            tree_dict = dict()
            with open(label_file_path, 'r') as file_stream:
                xml_tree = file_stream.readlines()
                xml_tree = ''.join([line.strip('\t') for line in xml_tree])
                xml_tree = BeautifulSoup(xml_tree, "html5lib")
                for obj in xml_tree.findAll('object'):
                    object = dict()
                    for name_tag in obj.findChildren('name'):
                        name = str(name_tag.contents[0])
                        difficult = int(obj.find('difficult').contents[0])
                        if name in CAT_DICT:
                            bbox = obj.findChildren('bndbox')[0]
                            xmin = int(float(bbox.findChildren('xmin')[0].contents[0]))
                            ymin = int(float(bbox.findChildren('ymin')[0].contents[0]))
                            xmax = int(float(bbox.findChildren('xmax')[0].contents[0]))
                            ymax = int(float(bbox.findChildren('ymax')[0].contents[0]))
                            object['bbox'] = [xmin, ymin, xmax, ymax]
                            object['label'] = CAT_DICT[name]
                            object['difficult'] = difficult
                            object_list.append(object)

            if len(object_list) == 0:
                continue

            tree_dict['objects'] = object_list
            if file_count % self.args.val_interval == 0:
                fw = open(os.path.join(self.val_json_dir, '{}.json'.format(label_file.split('.')[0])), 'w')
                fw.write(json.dumps(tree_dict))
                fw.close()
                shutil.copy(os.path.join(self.args.ori_img_dir, '{}.jpg'.format(label_file.split('.')[0])),
                            os.path.join(self.val_image_dir, '{}.jpg'.format(label_file.split('.')[0])))
            else:
                fw = open(os.path.join(self.train_json_dir, '{}.json'.format(label_file.split('.')[0])), 'w')
                fw.write(json.dumps(tree_dict))
                fw.close()
                shutil.copy(os.path.join(self.args.ori_img_dir, '{}.jpg'.format(label_file.split('.')[0])),
                            os.path.join(self.train_image_dir, '{}.jpg'.format(label_file.split('.')[0])))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_img_dir', default=None, type=str,
                        dest='ori_img_dir', help='The directory of the image data.')
    parser.add_argument('--ori_label_dir', default=None, type=str,
                        dest='ori_label_dir', help='The directory of the label data.')
    parser.add_argument('--val_interval', default=10, type=float,
                        dest='val_interval', help='The ratio of train & val data.')

    args = parser.parse_args()

    voc_det_generator = VocDetGenerator(args)
    voc_det_generator.generate_label()
