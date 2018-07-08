#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


ORI_ROOT_DIR='/data/DataSet/VOC'
SAVE_DIR='/data/DataSet/VOC_SEG'


function generate_label()
{
    if [! -d "$ORI_ROOT_DIR"]; then
        echo "Data not exists!!!"
        exit 0
    fi

    mkdir -p "${SAVE_DIR}/train/image"
    mkdir -p "${SAVE_DIR}/train/label"
    mkdir -p "${SAVE_DIR}/val/image"
    mkdir -p "${SAVE_DIR}/val/label"

    
    cp "${ORI_ROOT_DIR}/SegmentClass"
}



