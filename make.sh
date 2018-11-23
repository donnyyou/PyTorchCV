#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)

pip install -r requirements.txt

python setup.py develop

cd extensions/apis/cocoapi/PythonAPI
python setup.py install

cd -
cd extensions/layers/nms/src
make
rm -rf build