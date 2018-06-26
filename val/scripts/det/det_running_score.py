#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Object Detection running score.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class DetRunningScore(object):
    def __init__(self, configer):
        self.configer = configer
        self.fg_bg_AP = 0  # fg-bg map
        self.AP_array = np.zeros(self.configer.get('data', 'num_classes') - 1)
        self.cls_img_count = np.zeros(self.configer.get('data', 'num_classes') - 1)  # img numbers of each class
        self.img_count = 0

    def compute_ap(self, gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores, iou_threshold=0.5):
        """Compute Average Precision at a set IoU threshold (default 0.5).

        Returns:
        mAP: Mean Average Precision
        precisions: List of precisions at different class score thresholds.
        recalls: List of recall values at different class score thresholds.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
        """
        # Trim zero padding and sort predictions by score from high to low
        # TODO: cleaner to do zero unpadding upstream

        gt_boxes = self.trim_zeros(gt_boxes)
        pred_boxes = self._trim_zeros(pred_boxes)
        pred_scores = pred_scores[:pred_boxes.shape[0]]
        indices = np.argsort(pred_scores)[::-1]  # top2bottom
        pred_boxes = pred_boxes[indices]
        pred_class_ids = pred_class_ids[indices]

        # Compute IoU overlaps [pred_boxes, gt_boxes]
        overlaps = self.compute_overlaps(pred_boxes, gt_boxes)

        # Loop through ground truth boxes and find matching predictions
        match_count = 0
        pred_match = np.zeros([pred_boxes.shape[0]])
        gt_match = np.zeros([gt_boxes.shape[0]])  # tags:if a gt is matched,set its tag to 1;
        for i in range(len(pred_boxes)):
            # Find best matching ground truth box
            sorted_ixs = np.argsort(overlaps[i])[::-1]  # top2bottom
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                if gt_match[j] == 1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_threshold:
                    break
                # Do we have a match?
                if pred_class_ids[i] == gt_class_ids[j]:
                    match_count += 1
                    gt_match[j] = 1
                    pred_match[i] = 1
                    break

        # Compute precision and recall at each prediction box step
        precisions = np.cumsum(pred_match).astype(np.float32) / (np.arange(len(pred_match)) + 1)
        recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

        # Pad with start and end values to simplify the math
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])

        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])

        # Compute mean AP over recall range
        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

        return mAP, precisions, recalls, overlaps

    def compute_overlaps(self, boxes1, boxes2):
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].

        For better performance, pass the largest set first and the smaller second.
        """
        # Areas of anchors and GT boxes
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i]
            overlaps[:, i] = self._compute_iou(box2, boxes1, area2[i], area1)

        return overlaps

    def _compute_iou(self, box, boxes, box_area, boxes_area):
        """Calculates IoU of the given box with the array of the given boxes.
        box: 1D vector [y1, x1, y2, x2]
        boxes: [boxes_count, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.

        Note: the areas are passed in rather than calculated here for
              efficency. Calculate once in the caller to avoid duplicate work.
        """
        # Calculate intersection areas
        y1 = np.maximum(box[0], boxes[:, 0])
        y2 = np.minimum(box[2], boxes[:, 2])
        x1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box_area + boxes_area[:] - intersection[:]
        iou = intersection / union
        return iou

    def trim_zeros(self, x):
        """It's common to have tensors larger than the available data and
        pad with zeros. This function removes rows that are all zeros.

        x: [rows, columns].
        """
        # pdb.set_trace()
        assert len(x.shape) == 2
        return x[~np.all(x == 0, axis=1)]

    def update(self, batch_pred_bboxes, batch_gt_bboxes):
        """Evaluate predicted_file and return mAP."""
        # Construct set to speed up id searching.
        # for every annotation in our test/validation set
        for i in range(len(batch_pred_bboxes)):
            gt_boxes = np.array(batch_gt_bboxes[i][0:4])
            gt_class_ids = np.array(batch_gt_bboxes[i][4])
            pred_boxes = np.array(batch_pred_bboxes[i][0:4])
            pred_class_ids = np.array(batch_pred_bboxes[i][4])
            pred_scores = np.array(batch_pred_bboxes[i][5])
            img_mAP, _, _, _ = self.compute_ap(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores)
            self.fg_bg_AP = self.fg_bg_AP + img_mAP
            #
            # pdb.set_trace()
            cls_num = 0
            for cls in range(self.configer.get('data', 'num_classes') - 1):
                if cls in gt_class_ids:
                    cls_num = cls_num + 1
                    # pdb.set_trace()
                    idx = np.where(gt_class_ids == cls)[0]
                    pt_idx = np.where(pred_class_ids == cls)[0]

                    gt_b = gt_boxes[idx]
                    gt_c_ids = gt_class_ids[idx]
                    pt_b = pred_boxes[pt_idx]
                    pt_c_ids = pred_class_ids[pt_idx]
                    pt_s = pred_scores[pt_idx]

                    cls_AP, _, _, _ = self.compute_ap(gt_b, gt_c_ids, pt_b, pt_c_ids, pt_s)
                    # pdb.set_trace()
                    self.AP_array[cls] = self.AP_array[cls] + cls_AP
                    self.cls_img_count[cls] = self.cls_img_count[cls] + 1

            self.img_count += 1

    def get_mAP(self):
        # compute mAP by APs under different oks thresholds
        self.AP_array = self.AP_array[self.cls_img_count != 0] / self.cls_img_count[self.cls_img_count != 0]
        mAP = sum(self.AP_array) / len(self.AP_array)
        return mAP

    def get_fg_bg_AP(self):
        fg_bg_AP = self.fg_bg_AP / self.img_count
        return fg_bg_AP

    def reset(self):
        self.fg_bg_AP = 0  # fg-bg map
        self.AP_array = np.zeros(self.configer.get('data', 'num_classes') - 1)
        self.cls_img_count = np.zeros(self.configer.get('data', 'num_classes') - 1)  # img numbers of each class
        self.img_count = 0