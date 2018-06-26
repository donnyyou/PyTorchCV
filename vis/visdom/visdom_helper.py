#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualize the log files.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import visdom
import time
import numpy as np
import torchvision as tv

from utils.tools.logger import Logger as Log


class VisdomHelper():
    '''
      Repackage the package visdom.
    '''

    def __init__(self, env='default', **kwargs):

        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.global_win_dict = dict()
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot_line(self, win_name, line_name, x, y):
        '''
        self.plot('loss',1.00)
        '''
        if win_name not in self.global_win_dict:
            self.global_win_dict[win_name] = self.vis.line(X=np.array([x]),
                                                              Y=np.array([y]),
                                                              name=line_name,
                                                              opts={'legend': [line_name]})

        else:
            self.vis.updateTrace(X=np.array([x]), Y=np.array([y]),
                                    win=self.global_win_dict[win_name], name=line_name)

    def img(self, name, img_):
        '''
        self.img('input_img',t.Tensor(64,64))
        '''

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=unicode(name),
                       opts=dict(title=name)
                       )

    def img_grid_many(self, d):
        for k, v in d.iteritems():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        '''
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        '''
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win='log_text')

    def __getattr__(self, name):
        return getattr(self.vis, name)