#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Repackage some file operations.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from utils.tools.logger import Logger as Log


class FileHelper(object):

    @staticmethod
    def dir_name(file_path):
        return os.path.dirname(file_path)

    @staticmethod
    def abs_path(file_path):
        return os.path.abspath(file_path)

    @staticmethod
    def shotname(file_name):
        shotname, extension = os.path.splitext(file_name)
        return shotname

    @staticmethod
    def list_dir(dir_name, prefix=''):
        filename_list = list()
        items = os.listdir(os.path.join(dir_name, prefix))
        for item in items:
            fi_d = os.path.join(dir_name, prefix, item)
            if os.path.isdir(fi_d):
                prefix_temp = '{}/{}'.format(prefix, item).lstrip('/')
                filename_list += FileHelper.list_dir(dir_name, prefix_temp)
            else:
                filename_list.append('{}/{}'.format(prefix, item).lstrip('/'))

        return filename_list


if __name__ == "__main__":
    print (FileHelper.list_dir('/home/donny/Projects'))