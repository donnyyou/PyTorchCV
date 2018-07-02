#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Utilizer class for dataset loader.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


class DetDataUtilizer(object):

    def __init__(self, configer):
        self.configer = configer

    def _iou(self, box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

        Args:
          box1(tensor): bounding boxes, sized [N,4]; [[xmin, ymin, xmax, ymax], ...]
          box2(tensor): bounding boxes, sized [M,4].
        Return:
          iou(tensor): sized [N,M].

        """
        N = box1.size(0)
        M = box2.size(0)

        # max(xmin, ymin).
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
        )

        # min(xmax, ymax)
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def rpn_batch_encode(self, gt_bboxes, gt_labels, default_boxes):
        pass

    def rpn_item_encode(self, gt_bboxes, gt_labels, default_boxes):
        pass

    def roi_batch_encode(self, gt_bboxes, gt_labels, rois):
        pass

    def ssd_batch_encode(self, gt_bboxes, gt_labels, default_boxes):
        target_bboxes = list()
        target_labels = list()
        for i in range(len(gt_bboxes)):
            loc, conf = self.ssd_item_encode(gt_bboxes[i], gt_labels[i], default_boxes)
            target_bboxes.append(loc)
            target_labels.append(conf)

        return torch.stack(target_bboxes, 0), torch.stack(target_labels, 0)

    def ssd_item_encode(self, gt_bboxes, gt_labels, default_boxes):
        """Transform target bounding boxes and class labels to SSD boxes and classes.

        Match each object box to all the default boxes, pick the ones with the Jaccard-Index > threshold:
        Jaccard(A,B) = AB / (A+B-AB)

        Args:
          boxes(tensor): object bounding boxes (xmin,ymin,xmax,ymax) of a image, sized [#obj, 4].
          classes(tensor): object class labels of a image, sized [#obj,].
          threshold(float): Jaccard index threshold
        Returns:
          boxes(tensor): bounding boxes, sized [#obj, 8732, 4].
          classes(tensor): class labels, sized [8732,]

        """
        if gt_bboxes is None or len(gt_bboxes) == 0:
            loc = torch.zeros_like(default_boxes.size)
            conf = torch.zeros((default_boxes.size(0), )).long()

            return loc, conf

        iou = self._iou(gt_bboxes, torch.cat([default_boxes[:, :2] - default_boxes[:, 2:]/2,
                                              default_boxes[:, :2] + default_boxes[:, 2:]/2], 1))  # [#obj,8732]

        prior_box_iou, max_idx = iou.max(0)  # [1,8732]
        max_idx.squeeze_(0)  # [8732,]
        prior_box_iou.squeeze_(0)  # [8732,]

        boxes = gt_bboxes[max_idx]  # [8732,4]
        variances = [0.1, 0.2]
        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]  # [8732,2]
        cxcy /= variances[0] * default_boxes[:, 2:]
        wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]  # [8732,2]
        wh = torch.log(wh) / variances[1]
        loc = torch.cat([cxcy, wh], 1)  # [8732,4]

        conf = 1 + gt_labels[max_idx]  # [8732,], background class = 0

        if self.configer.get('details', 'anchor_method') == 'retina':
            conf[prior_box_iou < self.configer.get('details', 'iou_threshold')] = -1
            conf[prior_box_iou < self.configer.get('details', 'iou_threshold') - 0.1] = 0
        else:
            conf[prior_box_iou < self.configer.get('details', 'iou_threshold')] = 0  # background

        # According to IOU, it give every prior box a class label.
        # Then if the IOU is lower than the threshold, the class label is 0(background).
        class_iou, prior_box_idx = iou.max(1, keepdim=False)
        conf_class_idx = prior_box_idx.cpu().numpy()
        conf[conf_class_idx] = gt_labels + 1

        return loc, conf

