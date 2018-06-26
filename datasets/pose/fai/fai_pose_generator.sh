#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/deepmotion/PycharmProjects/PoseEstimation'

FAI_DATA_DIR='/home/deepmotion/DataSet/FashionAI/zip_file/train'

COCO_ANNO_DIR='/home/deepmotion/DataSet/FashionAI/zip_file/train/Annotations/'
TRAIN_ANNO_FILE=${COCO_ANNO_DIR}'train.csv'
VAL_ANNO_FILE=${COCO_ANNO_DIR}'annotations.csv'

TRAIN_ROOT_DIR='/home/deepmotion/DataSet/FashionAI/train'
VAL_ROOT_DIR='/home/deepmotion/DataSet/FashionAI/val'


python2.7 fai_pose_generator.py --root_dir $TRAIN_ROOT_DIR \
                                --anno_file $TRAIN_ANNO_FILE \
                                --data_dir $FAI_DATA_DIR \

python2.7 fai_pose_generator.py --root_dir $VAL_ROOT_DIR \
                                --anno_file $VAL_ANNO_FILE \
                                --data_dir $FAI_DATA_DIR
