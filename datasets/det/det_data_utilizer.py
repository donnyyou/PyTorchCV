#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Utilizer class for dataset loader.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch

from utils.helpers.det_helper import DetHelper
from utils.tools.logger import Logger as Log


class DetDataUtilizer(object):

    def __init__(self, configer):
        self.configer = configer

    def rpn_batch_encode(self, gt_bboxes, anchor_boxes):
        n_sample = self.configer.get('rpn', 'loss')['n_sample']
        pos_iou_thresh = self.configer.get('rpn', 'loss')['pos_iou_thresh']
        neg_iou_thresh = self.configer.get('rpn', 'loss')['neg_iou_thresh']
        pos_ratio = self.configer.get('rpn', 'loss')['pos_ratio']
        # Calc indicies of anchors which are located completely inside of the image
        # whose size is speficied.

        index_inside = (((anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2) >= 0)
                        & ((anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2) >= 0)
                        & ((anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2) <= 1.0)
                        & ((anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2) <= 1.0)).nonzero().contiguous().view(-1,)
        default_boxes = anchor_boxes[index_inside]
        target_bboxes = list()
        target_labels = list()
        for i in range(len(gt_bboxes)):
            loc = torch.zeros_like(default_boxes)
            label = torch.ones((default_boxes.size(0),)).mul_(-1).long()

            if gt_bboxes[i].numel() > 0:
                # label: 1 is positive, 0 is negative, -1 is dont care
                ious = DetHelper.bbox_iou(gt_bboxes[i],
                                          torch.cat([default_boxes[:, :2] - default_boxes[:, 2:] / 2,
                                                     default_boxes[:, :2] + default_boxes[:, 2:] / 2], 1))

                max_ious, argmax_ious = ious.max(0, keepdim=False)
                _, gt_argmax_ious = ious.max(1, keepdim=False)

                # assign negative labels first so that positive labels can clobber them
                label[max_ious < neg_iou_thresh] = 0

                # positive label: for each gt, anchor with highest iou
                label[gt_argmax_ious] = 1

                # positive label: above threshold IOU
                label[max_ious >= pos_iou_thresh] = 1

                # subsample positive labels if we have too many
                n_pos = int(pos_ratio * n_sample)
                pos_index = (label == 1).nonzero().contiguous().view(-1,).numpy()
                if len(pos_index) > n_pos:
                    disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
                    label[disable_index] = -1

                # subsample negative labels if we have too many
                n_neg = n_sample - torch.sum(label == 1).item()
                neg_index = (label == 0).nonzero().contiguous().view(-1,).numpy()

                if len(neg_index) > n_neg:
                    disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
                    label[disable_index] = -1

                boxes = gt_bboxes[i][argmax_ious]  # [8732,4]
                cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]  # [8732,2]
                cxcy /= default_boxes[:, 2:]
                wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]   # [8732,2]
                wh = torch.log(wh)
                loc = torch.cat([cxcy, wh], 1)  # [8732,4]
                # loc = loc[:, [1, 0, 3, 2]]
            else:
                # subsample negative labels if we have too many
                n_neg = n_sample // 2
                neg_index = (label == -1).nonzero().contiguous().view(-1, ).numpy()
                if len(neg_index) > n_neg:
                    disable_index = np.random.choice(neg_index, size=n_neg, replace=False)
                    label[disable_index] = 0

            ret_label = torch.ones((anchor_boxes.size(0),), dtype=torch.long).mul_(-1)
            ret_label[index_inside] = torch.LongTensor(label)
            ret_loc = torch.zeros((anchor_boxes.size(0), 4))
            ret_loc[index_inside] = loc
            target_bboxes.append(ret_loc)
            target_labels.append(ret_label)

        return torch.stack(target_bboxes, 0), torch.stack(target_labels, 0)

    def ssd_batch_encode(self, gt_bboxes, gt_labels, default_boxes):
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
        target_bboxes = list()
        target_labels = list()
        for i in range(len(gt_bboxes)):
            if gt_bboxes[i] is None or len(gt_bboxes[i]) == 0:
                loc = torch.zeros_like(default_boxes)
                conf = torch.zeros((default_boxes.size(0),)).long()

            else:

                iou = DetHelper.bbox_iou(gt_bboxes[i],
                                        torch.cat([default_boxes[:, :2] - default_boxes[:, 2:] / 2,
                                                   default_boxes[:, :2] + default_boxes[:, 2:] / 2], 1))  # [#obj,8732]

                prior_box_iou, max_idx = iou.max(0, keepdim=False)  # [1,8732]

                boxes = gt_bboxes[i][max_idx]  # [8732,4]
                variances = [0.1, 0.2]
                cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]  # [8732,2]
                cxcy /= variances[0] * default_boxes[:, 2:]
                wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]  # [8732,2]
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

    def yolo_batch_encode(self, batch_gt_bboxes, batch_gt_labels):
        anchors_list = self.configer.get('gt', 'anchors_list')
        stride_list = self.configer.get('network', 'stride_list')
        img_size = self.configer.get('data', 'input_size')

        assert len(anchors_list) == len(stride_list)
        batch_target_list = list()
        batch_objmask_list = list()
        batch_noobjmask_list = list()
        for fm_stride, ori_anchors in zip(stride_list, anchors_list):
            in_w, in_h = int(round(img_size[0] / fm_stride)), int(round(img_size[1] / fm_stride))
            anchors = [(a_w / fm_stride, a_h / fm_stride) for a_w, a_h in ori_anchors]
            batch_size = len(batch_gt_bboxes)
            num_anchors = len(anchors)
            obj_mask = torch.zeros(batch_size, num_anchors, in_h, in_w)
            noobj_mask = torch.ones(batch_size, num_anchors, in_h, in_w)
            tx = torch.zeros(batch_size, num_anchors, in_h, in_w)
            ty = torch.zeros(batch_size, num_anchors, in_h, in_w)
            tw = torch.zeros(batch_size, num_anchors, in_h, in_w)
            th = torch.zeros(batch_size, num_anchors, in_h, in_w)
            tconf = torch.zeros(batch_size, num_anchors, in_h, in_w)
            tcls = torch.zeros(batch_size, num_anchors, in_h, in_w, self.configer.get('data', 'num_classes'))

            for b in range(batch_size):
                for t in range(batch_gt_bboxes[b].size(0)):
                    # Convert to position relative to box
                    gx = (batch_gt_bboxes[b][t, 0] + batch_gt_bboxes[b][t, 2]) / 2.0 * in_w
                    gy = (batch_gt_bboxes[b][t, 1] + batch_gt_bboxes[b][t, 3]) / 2.0 * in_h
                    gw = (batch_gt_bboxes[b][t, 2] - batch_gt_bboxes[b][t, 0]) * in_w
                    gh = (batch_gt_bboxes[b][t, 3] - batch_gt_bboxes[b][t, 1]) * in_h
                    if gw * gh == 0 or gx >= in_w or gy >= in_h:
                        continue

                    # Get grid box indices
                    gi = int(gx)
                    gj = int(gy)
                    # Get shape of gt box
                    gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                    # Get shape of anchor box
                    anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((num_anchors, 2)),
                                                                      np.array(anchors)), 1))
                    # Calculate iou between gt and anchor shapes
                    anch_ious = DetHelper.bbox_iou(gt_box, anchor_shapes)
                    # Where the overlap is larger than threshold set mask to zero (ignore)
                    noobj_mask[b, anch_ious[0] > self.configer.get('gt', 'iou_threshold')] = 0
                    # Find the best matching anchor box
                    best_n = np.argmax(anch_ious, axis=1)

                    # Masks
                    obj_mask[b, best_n, gj, gi] = 1
                    # Coordinates
                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    # Width and height
                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
                    # object
                    tconf[b, best_n, gj, gi] = 1
                    # One-hot encoding of label
                    tcls[b, best_n, gj, gi, int(batch_gt_labels[b][t])] = 1

            obj_mask = obj_mask.view(batch_size, -1)
            noobj_mask = noobj_mask.view(batch_size, -1)
            tx = tx.view(batch_size, -1).unsqueeze(2)
            ty = ty.view(batch_size, -1).unsqueeze(2)
            tw = tw.view(batch_size, -1).unsqueeze(2)
            th = th.view(batch_size, -1).unsqueeze(2)
            tconf = tconf.view(batch_size, -1).unsqueeze(2)
            tcls = tcls.view(batch_size, -1, self.configer.get('data', 'num_classes'))
            target = torch.cat((tx, ty, tw, th, tconf, tcls), -1)
            batch_target_list.append(target)
            batch_objmask_list.append(obj_mask)
            batch_noobjmask_list.append(noobj_mask)

        return torch.cat(batch_target_list, 1), torch.cat(batch_objmask_list, 1), torch.cat(batch_noobjmask_list, 1)

