#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# TinyDB version Configer class for all hyper parameters.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os
import time
import argparse
from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

from utils.tools.logger import Logger as Log


class Configer(object):

    def __init__(self, args_parser=None, hypes_file=None, config_dict=None):
        self.tinydb = TinyDB('/tmp/db.json', storage=CachingMiddleware(JSONStorage))
        self.tinydb.purge()
        self.db_query = Query()
        params_root = None
        if config_dict is not None:
            params_root = config_dict

        elif hypes_file is not None:
            if not os.path.exists(hypes_file):
                Log.error('Json Path:{} not exists!'.format(hypes_file))
                exit(0)

            json_stream = open(hypes_file, 'r')
            params_root = json.load(json_stream)
            json_stream.close()

        elif args_parser is not None:
            args_dict = args_parser.__dict__
            if not os.path.exists(args_parser.hypes):
                print('Json Path:{} not exists!'.format(args_parser.hypes))
                exit(1)

            json_stream = open(args_parser.hypes, 'r')
            params_root = json.load(json_stream)
            json_stream.close()

            for key, value in args_dict.items():
                if Configer._is_empty(key.split(':'), params_root):
                    params_root = Configer._add_value(key.split(':'), value, params_root)

                elif value is not None:
                    params_root = Configer._update_value(key.split(':'), value, params_root)

        else:
            Log.error('Args ERROR!')
            exit(1)

        self._sync_db(params_root)

    def _sync_db(self, params_root):
        for key, value in params_root.items():
            if type(value) == dict:
                for sub_key, sub_value in value.items():
                    self.tinydb.insert({'key': '{}:{}'.format(key, sub_key), 'value': sub_value})
            else:
                self.tinydb.insert({'key': key, 'value': value})

        print(self.tinydb.all())

    @staticmethod
    def _get_caller():
        filename = os.path.basename(sys._getframe().f_back.f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_back.f_lineno
        prefix = '{}, {}'.format(filename, lineno)
        return prefix

    @staticmethod
    def _is_empty(key, params_root):
        if len(key) == 0:
            return True

        if len(key) == 1 and key[0] not in params_root:
            return True

        if len(key) == 2 and (key[0] not in params_root or key[1] not in params_root[key[0]]):
            return True

        return False

    @staticmethod
    def _add_value(key_tuple, value, params_root):
        if len(key_tuple) == 1 and key_tuple[0] not in params_root:
            params_root[key_tuple[0]] = value

        elif len(key_tuple) == 2:
            if key_tuple[0] not in params_root:
                params_root[key_tuple[0]] = dict()
                params_root[key_tuple[0]][key_tuple[1]] = value

            elif key_tuple[1] not in params_root[key_tuple[0]]:
                params_root[key_tuple[0]][key_tuple[1]] = value

            else:
                Log.error('{} Key: {} existed!!!'.format(Configer._get_caller(), key_tuple))
                exit(1)
        else:
            Log.error('{} KeyError: {}.'.format(Configer._get_caller(), key_tuple))
            exit(1)

        return params_root

    @staticmethod
    def _update_value(key_tuple, value, params_root):
        if len(key_tuple) == 1:
            if key_tuple[0] not in params_root:
                Log.error('{} Key: {} not existed!!!'.format(Configer._get_caller(), key_tuple))
                exit(1)

            params_root[key_tuple[0]] = value

        elif len(key_tuple) == 2:
            if key_tuple[0] not in params_root or key_tuple[1] not in params_root[key_tuple[0]]:
                Log.error('{} Key: {} not existed!!!'.format(Configer._get_caller(), key_tuple))
                exit(1)

            else:
                params_root[key_tuple[0]][key_tuple[1]] = value

        else:
            Log.error('{} Key: {} not existed!!!'.format(Configer._get_caller(), key_tuple))
            exit(1)

        return params_root

    def add_key_value(self, key_tuple, value):
        if not self.is_empty(*key_tuple):
            Log.error('{} Key: {} existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

        assert len(key_tuple) > 0
        query_key = ':'.join(key_tuple)
        self.tinydb.insert({'key': query_key, 'value': value})

    def update_value(self, key_tuple, value):
        # self.tinydb.close()

        if self.is_empty(*key_tuple):
            Log.error('{} Key: {} not existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

        query_key = ':'.join(key_tuple)
        print(self.tinydb.search(self.db_query.key == query_key)[0]['value'])
        self.tinydb.update({'value': value}, self.db_query.key == query_key)
        print(self.tinydb.search(self.db_query.key == query_key)[0]['value'])

    def plus_one(self, *key):
        if self.is_empty(*key):
            Log.error('{} Key: {} not existed!!!'.format(self._get_caller(), key))
            exit(1)

        query_key = ':'.join(key)
        v = self.tinydb.search(self.db_query.key == query_key)[0]['value']
        self.tinydb.update({'value': v + 1}, self.db_query.key == query_key)

    def get(self, *key):
        if self.is_empty(*key):
            Log.error('{} KeyError: {}.'.format(self._get_caller(), key))
            exit(1)

        query_key = ':'.join(key)
        out = self.tinydb.search(self.db_query.key == query_key)

        return out[0]['value']

    def is_empty(self, *key):
        query_key = ':'.join(key)
        if len(key) == 0:
            return True

        # print(self.tinydb.search(db_query.key == query_key))
        return len(self.tinydb.search(self.db_query.key == query_key)) != 1

    def to_dict(self):
        return self.tinydb.all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', default='../../hypes/cls/flower/fc_flower_cls.json', type=str,
                        dest='hypes', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of Pose Estimator.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of pretrained model.')
    parser.add_argument('--train_dir', default=None, type=str,
                        dest='data:train_dir', help='The path of train data.')

    args_parser = parser.parse_args()

    configer = Configer(args_parser=args_parser)

    configer.add_key_value(('project_dir',), 'root')
    configer.update_value(('project_dir', ), 'root1')

    print (configer.get('project_dir'))
    start_time = time.time()
    for i in range(10000):
        configer.get('project_dir')
        configer.get('network', 'resume')
        configer.get('logging', 'log_file')
        configer.get('data', 'train_dir')

    print (time.time() - start_time)

    start_time = time.time()
    for i in range(30000):
        json_stream = open(args_parser.hypes, 'r')
        params_root = json.load(json_stream)

    print(time.time() - start_time)