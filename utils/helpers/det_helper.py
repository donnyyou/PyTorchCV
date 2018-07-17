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

