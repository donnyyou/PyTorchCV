#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


class DetHelper(object):

    @staticmethod
    def nms(bboxes, scores=None, nms_threshold=0.0, mode='union'):
        """Non maximum suppression.

        Args:
          bboxes(tensor): bounding boxes, sized [N,4].
          scores(tensor): bbox scores, sized [N,].
          threshold(float): overlap threshold.
          mode(str): 'union' or 'min'.

        Returns:
          keep(tensor): selected indices.

        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        """
        bboxes = bboxes.contiguous().view(-1, 4)
        if scores is not None:
            scores = scores.contiguous().view(-1,)

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        if scores is not None:
            _, order = scores.sort(0, descending=True)
        else:
            order = np.arange(len(bboxes), dtype=np.int32)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0]
            keep.append(i)
            xx1 = x1[order[1:]].clamp(min=x1[i].item())
            yy1 = y1[order[1:]].clamp(min=y1[i].item())
            xx2 = x2[order[1:]].clamp(max=x2[i].item())
            yy2 = y2[order[1:]].clamp(max=y2[i].item())

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= nms_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break

            order = order[ids + 1]

        return torch.LongTensor(keep)

    @staticmethod
    def cls_nms(bboxes, scores=None, labels=None, nms_threshold=0.0, mode='union'):
        unique_labels = labels.cpu().unique()
        bboxes = bboxes.contiguous().view(-1, 4)
        if scores is not None:
            scores = scores.contiguous().view(-1,)

        if labels is not None:
            labels = labels.contiguous().view(-1,)

        if bboxes.is_cuda:
            unique_labels = unique_labels.cuda()

        cls_keep_list = list()
        for c in unique_labels:
            cls_index = torch.nonzero(labels == c).squeeze(1)
            if scores is not None:
                cls_keep = DetHelper.nms(bboxes[cls_index],
                                         scores=scores[cls_index],
                                         nms_threshold=nms_threshold,
                                         mode=mode)
            else:
                cls_keep = DetHelper.nms(bboxes[cls_index],
                                         nms_threshold=nms_threshold,
                                         mode=mode)

            cls_keep_list.append(cls_index[cls_keep])

        return torch.cat(cls_keep_list, 0)

    @staticmethod
    def bbox_iou(box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

        Args:
          box1(tensor): bounding boxes, sized [N,4]; [[xmin, ymin, xmax, ymax], ...]
          box2(tensor): bounding boxes, sized [M,4].
        Return:
          iou(tensor): sized [N,M].

        """
        if len(box1.size()) == 1:
            box1 = box1.unsqueeze(0)

        if len(box2.size()) == 1:
            box2 = box2.unsqueeze(0)

        N = box1.size(0)
        M = box2.size(0)

        # max(xmin, ymin).
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
        )

        # min(xmax, ymax)
        rb = torch.min(
            box1[:, 2:4].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:4].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
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

    @staticmethod
    def bbox_kmeans(bboxes, cluster_number, dist=np.median):
        box_number = bboxes.shape[0]
        distances = np.empty((box_number, cluster_number))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = bboxes[np.random.choice(
            box_number, cluster_number, replace=False)]  # init k clusters
        while True:

            distances = 1 - DetHelper.iou(bboxes, clusters, cluster_number)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(cluster_number):
                clusters[cluster] = dist(  # update clusters
                    bboxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        result = clusters[np.lexsort(clusters.T[0, None])]
        avg_iou = DetHelper.avg_iou(bboxes, result, cluster_number)
        return result, avg_iou

    @staticmethod
    def iou(boxes, clusters, k):  # 1 box -> k clusters
        n = boxes.shape[0]

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    @staticmethod
    def avg_iou(boxes, clusters, k):
        accuracy = np.mean([np.max(DetHelper.iou(boxes, clusters, k), axis=1)])
        return accuracy


if __name__ == "__main__":
    cluster_number = 9
    filename = "2012_train.txt"
    kmeans = DetHelper.bbox_kmeans(None, None)

