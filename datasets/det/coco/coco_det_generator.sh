#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'


COCO_TRAIN_IMG_DIR='/home/donny/DataSet/MSCOCO/train2017'
COCO_VAL_IMG_DIR='/home/donny/DataSet/MSCOCO/val2017'

COCO_ANNO_DIR='/home/donny/DataSet/MSCOCO/annotations/'
TRAIN_ANNO_FILE=${COCO_ANNO_DIR}'instances_train2017.json'
VAL_ANNO_FILE=${COCO_ANNO_DIR}'instances_val2017.json'

TRAIN_SAVE_DIR='/home/donny/DataSet/COCO_DET/train'
VAL_SAVE_DIR='/home/donny/DataSet/COCO_DET/val'


python coco_det_generator.py --save_dir $TRAIN_SAVE_DIR \
                             --anno_file $TRAIN_ANNO_FILE \
                             --ori_img_dir $COCO_TRAIN_IMG_DIR

python coco_det_generator.py --save_dir $VAL_SAVE_DIR \
                             --anno_file $VAL_ANNO_FILE \
                             --ori_img_dir $COCO_VAL_IMG_DIR