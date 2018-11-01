#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)

pip3 install -r requirements.txt

python setup.py develop

cd extensions/apis/cocoapi/PythonAPI
python setup.py install