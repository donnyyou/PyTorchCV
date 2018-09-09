#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn.functional as F


class DataTransformer(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        size_mode: max_size, fix_size, random_size, multi_size
        scale_max: the max scale to resize.
    """

    def __init__(self, configer):
        self.configer = configer

    def __call__(self, img_list=None, labelmap_list=None, maskmap_list=None,
                 kpts_list=None, bboxes_list=None, labels_list=None,
                 polygons_list=None, trans_dict=None):

        if trans_dict['size_mode'] == 'random_size':
            return {
                'img': torch.stack(img_list, 0),
                'labelmap': None if labelmap_list is None else torch.stack(labelmap_list, 0),
                'maskmap': None if maskmap_list is None else torch.stack(maskmap_list, 0),
                'kpts': kpts_list,
                'bboxes': bboxes_list,
                'labels': labels_list,
                'polygons': polygons_list
            }

        if trans_dict['size_mode'] == 'fix_size':
            target_width, target_height = trans_dict['input_size']
        elif trans_dict['size_mode'] == 'multi_size':
            ms_input_size = trans_dict['ms_input_size']
            target_width, target_height = ms_input_size[random.randint(0, len(ms_input_size))]
        elif trans_dict['size_mode'] == 'max_size':
            border_width = [img.size(2) for img in img_list]
            border_height = [img.size(1) for img in img_list]
            target_width, target_height = max(border_width), max(border_height)
        else:
            raise NotImplementedError('Size Mode {} is invalid!'.format(trans_dict['size_mode']))

        for i in range(len(img_list)):
            channels, height, width = img_list[i].size()
            scaled_size = [width, height]

            if trans_dict['align_method'] in ['only_scale', 'scale_and_pad']:
                w_scale_ratio = target_width / width
                h_scale_ratio = target_height / height
                if trans_dict['align_method'] == 'scale_and_pad':
                    w_scale_ratio = min(w_scale_ratio, h_scale_ratio)
                    h_scale_ratio = w_scale_ratio

                if kpts_list is not None and kpts_list[i].numel() > 0:
                    kpts_list[i][:, :, 0] *= w_scale_ratio
                    kpts_list[i][:, :, 1] *= h_scale_ratio

                if bboxes_list is not None and bboxes_list[i].numel() > 0:
                    bboxes_list[i][:, 0::2] *= w_scale_ratio
                    bboxes_list[i][:, 1::2] *= h_scale_ratio

                if polygons_list is not None:
                    for object_id in range(len(polygons_list)):
                        for polygon_id in range(len(polygons_list[object_id])):
                            polygons_list[i][object_id][polygon_id][0::2] *= w_scale_ratio
                            polygons_list[i][object_id][polygon_id][1::2] *= h_scale_ratio

                scaled_size = (int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio)))
                img_list[i] = F.interpolate(img_list[i].unsqueeze(0), scaled_size, mode='bilinear').squeeze(0)
                if labelmap_list is not None:
                    labelmap_list[i] = F.interpolate(labelmap_list[i].unsqueeze(0).unsqueeze(0).float(),
                                                     scaled_size, mode='nearest').long().squeeze(0).squeeze(0)

                if maskmap_list is not None:
                    maskmap_list[i] = F.interpolate(maskmap_list[i].unsqueeze(0).unsqueeze(0).float(),
                                                    scaled_size, mode='nearest').long().squeeze(0).squeeze(0)

            pad_width = target_width - scaled_size[0]
            pad_height = target_height - scaled_size[1]
            assert pad_height >= 0 and pad_width >= 0
            if pad_width > 0 or pad_height > 0:
                assert trans_dict['align_method'] in ['only_pad', 'scale_and_pad']
                left_pad = random.randint(0, pad_width)  # pad_left
                up_pad = random.randint(0, pad_height)  # pad_up

                expand_image = torch.zeros((channels, target_height, target_width))
                expand_image[:, up_pad:up_pad + height, left_pad:left_pad + width] = img_list[i]
                img_list[i] = expand_image

                if labelmap_list is not None:
                    expand_labelmap = torch.zeros((target_height, target_width)).long()
                    expand_labelmap[:, :] = self.configer.get('data', 'num_classes')
                    expand_labelmap[up_pad:up_pad + height, left_pad:left_pad + width] =labelmap_list[i]
                    labelmap_list[i] = expand_labelmap

                if maskmap_list is not None:
                    expand_maskmap = torch.ones((target_height, target_width)).long()
                    expand_maskmap[up_pad:up_pad + height, left_pad:left_pad + width] =maskmap_list[i]
                    maskmap_list[i] = expand_maskmap

                if polygons_list is not None:
                    for object_id in range(len(polygons_list)):
                        for polygon_id in range(len(polygons_list[object_id])):
                            polygons_list[i][object_id][polygon_id][0::2] += left_pad
                            polygons_list[i][object_id][polygon_id][1::2] += up_pad

                if kpts_list is not None and kpts_list[i].numel() > 0:
                    kpts_list[i][:, :, 0] += left_pad
                    kpts_list[i][:, :, 1] += up_pad

                if bboxes_list is not None and bboxes_list[i].numel() > 0:
                    bboxes_list[i][:, 0::2] += left_pad
                    bboxes_list[i][:, 1::2] += up_pad

        return {
            'img': torch.stack(img_list, 0),
            'labelmap': None if labelmap_list is None else torch.stack(labelmap_list, 0),
            'maskmap': None if maskmap_list is None else torch.stack(maskmap_list, 0),
            'kpts': kpts_list,
            'bboxes': bboxes_list,
            'labels': labels_list,
            'polygons': polygons_list
        }
