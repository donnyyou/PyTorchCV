#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhiwei Li, Kuiyuan Yang (zhiweili@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import numpy as np
import cv2
import arrows_gb2006


class Completor(object):
    def __init__(self):
        pass

    def __call__(self, arrow_name, obv_arrow, obv_id):
        std_arrow = arrows_gb2006.dict_arrow_name_to_polygon[arrow_name]

        used_std_keypoints = std_arrow[obv_id, :]
        used_obv_keypoints = obv_arrow

        H, status = cv2.findHomography(used_std_keypoints, used_obv_keypoints)

        std_arrow = np.hstack((std_arrow, np.ones((std_arrow.shape[0], 1))))

        proj = np.matmul(H, std_arrow.transpose())
        for idx in range(proj.shape[1]):
            proj[:, idx] /= proj[2, idx]
        return proj

if __name__ == '__main__':

    class CoordinateStore:
        def __init__(self):
            self.points = []

        def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print('click')
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
                self.points.append((x, y))


    completor = Completor()

    cv2.namedWindow('image')
    img = cv2.imread('./data/sample.jpg')
    coordinateStore1 = CoordinateStore()
    cv2.setMouseCallback('image', coordinateStore1.select_point)
    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('a'):
            break

    used = [0, 1, 2, 4, 5, 7, 8]
    proj_points = completor('arrow_left', np.array(coordinateStore1.points), used)

    proj_points = np.int32(proj_points[0:2, :].transpose())
    cv2.polylines(img, [proj_points], isClosed=True, color=(255, 0, 0), thickness=3)
    cv2.imshow('image', img)
    cv2.waitKey(0)
