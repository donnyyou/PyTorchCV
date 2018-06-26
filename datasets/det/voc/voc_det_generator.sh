#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'


ORI_IMG_DIR='/home/donny/DataSet/DeepFashion/img_selected/img'
ORI_LABEL_DIR='/home/donny/DataSet/DeepFashion/img_selected/anno'
SAVE_DIR='/home/donny/DataSet/Fashion1'


python2.7 voc_det_generator.py --ori_img_dir $ORI_IMG_DIR \
                               --ori_label_dir $ORI_LABEL_DIR \
                               --save_dir $SAVE_DIR \
                               --val_interval 10