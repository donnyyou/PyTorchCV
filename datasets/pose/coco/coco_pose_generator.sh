#!/usr/bin bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'

INPUT_SIZE=368

COCO_TRAIN_IMG_DIR='/data/DataSet/MSCOCO/train2017'
COCO_VAL_IMG_DIR='/data/DataSet/MSCOCO/val2017'

COCO_ANNO_DIR='/data/DataSet/MSCOCO/annotations/'
TRAIN_ANNO_FILE=${COCO_ANNO_DIR}'person_keypoints_train2017.json'
VAL_ANNO_FILE=${COCO_ANNO_DIR}'person_keypoints_val2017.json'

TRAIN_ROOT_DIR='/data/DataSet/COCO_MASK/train'
VAL_ROOT_DIR='/data/DataSet/COCO_MASK/val'


python2.7 coco_pose_generator.py --root_dir $TRAIN_ROOT_DIR \
                                 --ori_anno_file $TRAIN_ANNO_FILE \
                                 --ori_img_dir $COCO_TRAIN_IMG_DIR

python2.7 coco_pose_generator.py --root_dir $VAL_ROOT_DIR \
                                 --ori_anno_file $VAL_ANNO_FILE \
                                 --ori_img_dir $COCO_VAL_IMG_DIR
