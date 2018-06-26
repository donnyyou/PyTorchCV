#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhiwei Li (zhiweili@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import numpy as np


polygon_arrow_straight = np.array([             #coherent
    [45, 600],
    [ 0, 360],
    [30, 360],
    [30,   0],
    [60,   0],
    [60, 360],
    [90, 360]
], dtype=np.float32)
polygon_arrow_straight_robustness = np.array([1, 1, 1, 1, 1, 1, 1])

polygon_arrow_left = np.array([                 #coherent
    [0, 440],
    [40, 290],
    [40, 380],
    [120, 300],
    [120,   0],
    [150,   0],
    [150, 390],
    [40, 500],
    [40, 600]
], dtype=np.float32)
polygon_arrow_left_robustness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

polygon_arrow_right = np.array([                #coherent
    [150, 440],
    [110, 600],
    [110, 500],
    [0, 390],
    [0,   0],
    [30,   0],
    [30, 300],
    [110, 380],
    [110, 290]
])
polygon_arrow_right_robustness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

polygon_arrow_left_right = np.array([           #coherent
    [135, 405],
    [ 40, 500],
    [ 40, 600],
    [  0, 440],
    [ 40, 290],
    [ 40, 380],
    [120, 300],
    [120,   0],
    [150,   0],
    [150, 300],
    [230, 380],
    [230, 290],
    [270, 440],
    [230, 600],
    [230, 500]
])
polygon_arrow_left_right_robustness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

polygon_arrow_straight_left = np.array([        #coherent
    [135, 600],
    [ 90, 360],
    [120, 360],
    [120, 160],
    [ 40, 240],
    [ 40, 340],
    [  0, 180],
    [ 40,  30],
    [ 40, 120],
    [120,  40],
    [120,   0],
    [150,   0],
    [150, 360],
    [180, 360]
], dtype=np.float32)
polygon_arrow_straight_left_robustness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

polygon_arrow_straight_right = np.array([       #coherent
    [ 45, 600],
    [  0, 360],
    [ 30, 360],
    [ 30, 0],
    [ 60,   0],
    [ 60,  40],
    [140, 120],
    [140,  30],
    [180, 180],
    [140, 340],
    [140, 240],
    [ 60, 160],
    [ 60, 360],
    [ 90, 360]
])
polygon_arrow_straight_right_robustness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

polygon_arrow_straight_left_right = np.array([  #coherent
    [135, 600],
    [ 90, 360],
    [120, 360],
    [120, 160],
    [ 40, 240],
    [ 40, 340],
    [  0, 180],
    [ 40,  30],
    [ 40, 120],
    [120,  40],
    [120,   0],
    [150,   0],
    [150,  40],
    [230, 120],
    [230,  30],
    [270, 180],
    [230, 340],
    [230, 240],
    [150, 160],
    [150, 360],
    [180, 360]
])
polygon_arrow_straight_left_right_robustness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

polygon_arrow_uturn = np.vstack((      #coherent
    [ 60, 160],
    [120, 400],
    [ 80, 400],

    np.array([130, 460]) - [50, 0],   #inner half-circle, radius is 50
    np.array([130, 460]) + [0, 50],
    np.array([130, 460]) + [50, 0],

    [180,   0],
    [220,   0],

    np.array([130, 510]) + [90, 0],   #outer half-circle, radius is 90
    np.array([130, 510]) + [0, 90],
    np.array([130, 510]) - [90, 0],

    [ 40, 400],
    [  0, 400]
))
polygon_arrow_uturn_robustness = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])

outer_radius = 52.5
outer_theta = np.arccos((90.0 -  outer_radius) / outer_radius)
inner_radius = 30.0
polygon_arrow_straight_uturn = np.vstack((  #coherent
    [135, 600],
    [ 90, 360],
    [120, 360],

    outer_radius * np.array([np.cos(outer_theta), np.sin(outer_theta)]) + [30 + outer_radius, 295 - outer_radius],
    outer_radius * np.array([0, 1]) + [30 + outer_radius, 295 - outer_radius],
    outer_radius * np.array([-1, 0]) + [30 + outer_radius, 295 - outer_radius],

    [ 30, 180],
    [  0, 180],
    [ 45,   0],
    [ 90, 180],
    [ 60, 180],

    inner_radius * np.array([-1, 0]) + [ 90, 245 - inner_radius],   #only two points on the half circle
    inner_radius * np.array([1, 0]) + [ 90, 245 - inner_radius],

    [120,   0],
    [150,   0],
    [150, 360],
    [180, 360]
))
polygon_arrow_straight_uturn_robustness = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])

polygon_arrow_left_uturn = np.vstack((  #coherent
    [  0, 440],
    [ 40, 290],
    [ 40, 380],
    [120, 300],

    outer_radius * np.array([np.cos(outer_theta), np.sin(outer_theta)]) + [30 + outer_radius, 295 - outer_radius],
    outer_radius * np.array([0, 1]) + [30 + outer_radius, 295 - outer_radius],
    outer_radius * np.array([-1, 0]) + [30 + outer_radius, 295 - outer_radius],

    [ 30, 180],
    [  0, 180],
    [ 45,   0],
    [ 90, 180],
    [ 60, 180],

    inner_radius * np.array([-1, 0]) + [90, 245 - inner_radius],  # only two points on the half circle
    inner_radius * np.array([1, 0]) + [90, 245 - inner_radius],

    [120,   0],
    [150,   0],
    [150, 390],
    [ 40, 500],
    [ 40, 600]
))
polygon_arrow_left_uturn_robustness = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])

triangle_top_angle = 2 * np.arctan(45.0 / 240.0)
print('traingle_top_angle', triangle_top_angle * 180.0 / np.pi)
top_left_angle = 11.2 * np.pi / 180.0
triangle_waist_length = np.sqrt(45.0*45.0 + 240.0*240.0)
print('waist_length', triangle_waist_length)
triangle_bottom_left_vertex  = np.array([triangle_waist_length * np.sin(top_left_angle), 600 - triangle_waist_length * np.cos(top_left_angle)])
triangle_bottom_right_vertex = np.array([triangle_waist_length * np.sin(top_left_angle + triangle_top_angle), 600 - triangle_waist_length * np.cos(top_left_angle + triangle_top_angle)])
triangle_bottom_left_middle_vertex = triangle_bottom_left_vertex + (triangle_bottom_right_vertex - triangle_bottom_left_vertex) / 3.0
triangle_bottom_right_middle_vertex = triangle_bottom_left_vertex + (triangle_bottom_right_vertex - triangle_bottom_left_vertex) * 2.0 / 3.0

arrow_direction = np.array([-np.sin(top_left_angle + 0.5 * triangle_top_angle), np.cos(top_left_angle + 0.5 * triangle_top_angle)])
arrow_direction = arrow_direction / np.linalg.norm(arrow_direction)

polygon_arrow_merge_left = np.vstack((
    [0,   600],
    triangle_bottom_left_vertex,
    triangle_bottom_left_middle_vertex,

    [120, 180],
    [120,   0],
    [150,   0],
    [150, 180],

    triangle_bottom_right_middle_vertex,
    triangle_bottom_right_vertex,
))
polygon_arrow_merge_left_robustness = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1])

polygon_arrow_merge_right = np.copy(polygon_arrow_merge_left)
polygon_arrow_merge_right[:, 0] = 150.0 - polygon_arrow_merge_right[:, 0]
polygon_arrow_merge_right = np.flipud(polygon_arrow_merge_right)
polygon_arrow_merge_right_robustness = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1])

arrow_name_list = [
    'arrow_straight',
    'arrow_left',
    'arrow_right',
    'arrow_straight_left',
    'arrow_straight_right',
    'arrow_straight_left_right',
    'arrow_left_right',
    'arrow_uturn',
    'arrow_uturn_left',
    'arrow_uturn_straight',
    'arrow_merge_left',
    'arrow_merge_right'
]

dict_arrow_name_to_polygon = {
    'arrow_straight'            : polygon_arrow_straight,
    'arrow_left'                : polygon_arrow_left,
    'arrow_right'               : polygon_arrow_right,
    'arrow_straight_left'       : polygon_arrow_straight_left,
    'arrow_straight_right'      : polygon_arrow_straight_right,
    'arrow_straight_left_right' : polygon_arrow_straight_left_right,
    'arrow_uturn'               : polygon_arrow_uturn,
    'arrow_uturn_left'          : polygon_arrow_left_uturn,
    'arrow_uturn_straight'      : polygon_arrow_straight_uturn,
    'arrow_left_right'          : polygon_arrow_left_right,
    'arrow_merge_left'          : polygon_arrow_merge_left,
    'arrow_merge_right'         : polygon_arrow_merge_right
}

dict_arrow_name_to_robustness = {
    'arrow_straight'            : polygon_arrow_straight_robustness,
    'arrow_left'                : polygon_arrow_left_robustness,
    'arrow_right'               : polygon_arrow_right_robustness,
    'arrow_straight_left'       : polygon_arrow_straight_left_robustness,
    'arrow_straight_right'      : polygon_arrow_straight_right_robustness,
    'arrow_straight_left_right' : polygon_arrow_straight_left_right_robustness,
    'arrow_uturn'               : polygon_arrow_uturn_robustness,
    'arrow_uturn_left'          : polygon_arrow_left_uturn_robustness,
    'arrow_uturn_straight'      : polygon_arrow_straight_uturn_robustness,
    'arrow_left_right'          : polygon_arrow_left_right_robustness,
    'arrow_merge_left'          : polygon_arrow_merge_left_robustness,
    'arrow_merge_right'         : polygon_arrow_merge_right_robustness
}

if __name__ == '__main__':
    print('test')