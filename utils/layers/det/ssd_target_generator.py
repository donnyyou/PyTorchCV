#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch

from utils.helpers.det_helper import DetHelper
from utils.layers.det.ssd_priorbox_layer import SSDPriorBoxLayer
from utils.tools.logger import Logger as Log


class SSDTargetGenerator(object):
    """Compute prior boxes coordinates in center-offset form for each source feature map."""

    def __init__(self, configer):
        self.configer = configer
        self.fr_proirbox_layer = SSDPriorBoxLayer(configer)

    def __call__(self, feat_list, gt_bboxes, gt_labels, input_size):
        anchor_boxes = self.fr_proirbox_layer(feat_list, input_size)
        target_bboxes = list()
        target_labels = list()
        for i in range(len(gt_bboxes)):
            if gt_bboxes[i] is None or len(gt_bboxes[i]) == 0:
                loc = torch.zeros_like(anchor_boxes)
                conf = torch.zeros((anchor_boxes.size(0),)).long()

            else:

                iou = DetHelper.bbox_iou(gt_bboxes[i],
                                         torch.cat([anchor_boxes[:, :2] - anchor_boxes[:, 2:] / 2,
                                                    anchor_boxes[:, :2] + anchor_boxes[:, 2:] / 2], 1))  # [#obj,8732]

                prior_box_iou, max_idx = iou.max(0, keepdim=False)  # [1,8732]

                boxes = gt_bboxes[i][max_idx]  # [8732,4]
                variances = [0.1, 0.2]
                cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - anchor_boxes[:, :2]  # [8732,2]
                cxcy /= variances[0] * anchor_boxes[:, 2:]
                wh = (boxes[:, 2:] - boxes[:, :2]) / anchor_boxes[:, 2:]  # [8732,2]
                wh = torch.log(wh) / variances[1]
                loc = torch.cat([cxcy, wh], 1)  # [8732,4]

                conf = 1 + gt_labels[i][max_idx]  # [8732,], background class = 0

                if self.configer.get('gt', 'anchor_method') == 'retina':
                    conf[prior_box_iou < self.configer.get('gt', 'iou_threshold')] = -1
                    conf[prior_box_iou < self.configer.get('gt', 'iou_threshold') - 0.1] = 0
                else:
                    conf[prior_box_iou < self.configer.get('gt', 'iou_threshold')] = 0  # background

                # According to IOU, it give every prior box a class label.
                # Then if the IOU is lower than the threshold, the class label is 0(background).
                class_iou, prior_box_idx = iou.max(1, keepdim=False)
                conf_class_idx = prior_box_idx.cpu().numpy()
                conf[conf_class_idx] = gt_labels[i] + 1

            target_bboxes.append(loc)
            target_labels.append(conf)

        return torch.stack(target_bboxes, 0), torch.stack(target_labels, 0)