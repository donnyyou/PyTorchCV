#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class FRAnchorPredictLayer(nn.Module):
    def __init__(self, configer):
        super(FRAnchorPredictLayer, self).__init__()
        self.configer = configer
        self.num_anchor_list = self.configer.get('rpn', 'num_anchor_list')
        self.num_features = configer.get('rpn', 'num_feature_list')
        self.score = nn.Conv2d(self.num_features[0], self.num_anchor_list[0] * 2, 1, 1, 0)
        self.loc = nn.Conv2d(self.num_features[0], self.num_anchor_list[0] * 4, 1, 1, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat_list):
        rpn_locs = self.loc(feat_list[0])
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(feat_list[0].size(0), -1, 4)
        rpn_scores = self.score(feat_list[0])
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(feat_list[0].size(0), -1, 2)
        return rpn_locs, rpn_scores