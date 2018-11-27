#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
PYTHON=${PYTHON:-"python"}

pip install -r requirements.txt

cd extensions/apis/cocoapi/PythonAPI
python setup.py install

cd -
echo "Building roi align op..."
cd extensions/layers/roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms/src
make clean
make PYTHON=${PYTHON}