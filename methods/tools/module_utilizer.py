#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some methods used by main methods.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn

from extensions.layers.encoding.parallel import DataParallelModel
from utils.tools.logger import Logger as Log


class ModuleUtilizer(object):

    def __init__(self, configer):
        self.configer = configer
        self._init()

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if self.configer.get('network', 'init') == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight.data)

            elif self.configer.get('network', 'init') == 'xavier_normal':
                nn.init.xavier_normal_(m.weight.data)

            else:
                Log.error('Invalid init method {}'.format(self.configer.get('network', 'init')))
                exit(1)

    def _init(self):
        self.configer.add_key_value(['iters'], 0)
        self.configer.add_key_value(['last_iters'], 0)
        self.configer.add_key_value(['epoch'], 0)
        self.configer.add_key_value(['last_epoch'], 0)
        self.configer.add_key_value(['max_performance'], 0.0)
        self.configer.add_key_value(['performance'], 0.0)
        self.configer.add_key_value(['min_val_loss'], 9999.0)
        self.configer.add_key_value(['val_loss'], 9999.0)

    def to_device(self, *params):
        device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        return_list = list()
        for i in range(len(params)):
            return_list.append(params[i].to(device))

        return return_list

    def _make_parallel(self, net):
        if not self.configer.is_empty('network', 'encoding_parallel')\
                and self.configer.get('network', 'encoding_parallel'):
            return DataParallelModel(net)

        else:
            return nn.DataParallel(net)

    def load_net(self, net):
        net = self._make_parallel(net)
        net = net.to(torch.device('cpu' if self.configer.get('gpu') is None else 'cuda'))

        if self.configer.get('network', 'resume') is not None:
            if self.configer.get('network', 'resume_level') == 'full':
                checkpoint_dict = torch.load(self.configer.get('network', 'resume'))
                if 'state_dict' not in checkpoint_dict:
                    net.load_state_dict(checkpoint_dict)
                else:
                    net.load_state_dict(checkpoint_dict['state_dict'])

            elif self.configer.get('network', 'resume_level') == 'part':
                checkpoint_dict = torch.load(self.configer.get('network', 'resume'))
                load_dict = net.state_dict()
                pretrained_dict = dict()
                for key, value in checkpoint_dict['state_dict'].items():
                    if key.split('.')[0] == 'module':
                        key_in = key

                    else:
                        key_in = 'module.{}'.format(key)

                    if key_in in load_dict and load_dict[key_in].size() == value.size():
                        pretrained_dict[key_in] = checkpoint_dict['state_dict'][key]

                    else:
                        Log.info('Key {} is not match!'.format(key_in))

                load_dict.update(pretrained_dict)
                net.load_state_dict(load_dict)

        elif not self.configer.get('network', 'pretrained'):
            net.apply(self.__weights_init)

        return net

    def save_net(self, net, metric='iters'):
        state = {
            'config_dict': self.configer.to_dict(),
            'state_dict': net.state_dict(),
        }
        if self.configer.get('checkpoints', 'checkpoints_root') is None:
            checkpoints_dir = os.path.join(self.configer.get('project_dir'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))
        else:
            checkpoints_dir = os.path.join(self.configer.get('checkpoints', 'checkpoints_root'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        if metric == 'performance':
            if self.configer.get('performance') > self.configer.get('max_performance'):
                latest_name = '{}_max_performance.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update_value(['max_performance'], self.configer.get('performance'))

        elif metric == 'val_loss':
            if self.configer.get('val_loss') < self.configer.get('min_val_loss'):
                latest_name = '{}_min_loss.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update_value(['min_val_loss'], self.configer.get('val_loss'))

        elif metric == 'iters':
            if self.configer.get('iters') - self.configer.get('last_iters') >= \
                    self.configer.get('checkpoints', 'save_iters'):
                latest_name = '{}_iters{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('iters'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update_value(['last_iters'], self.configer.get('iters'))

        elif metric == 'epoch':
            if self.configer.get('epoch') - self.configer.get('last_epoch') >= \
                    self.configer.get('checkpoints', 'save_epoch'):
                latest_name = '{}_epoch{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('epoch'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update_value(['last_epoch'], self.configer.get('epoch'))

        else:
            Log.error('Metric: {} is invalid.'.format(metric))
            exit(1)

    def freeze_bn(self, net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


