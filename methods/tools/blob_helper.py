#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Generate the inputs.


import cv2
import numpy as np
from PIL import Image

from datasets.tools.transforms import DeNormalize
from datasets.tools.transforms import Scale
from utils.helpers.image_helper import ImageHelper


class BlobHelper(object):
    def __init__(self, configer):
        self.configer = configer

    def make_input_list(self, image_path):
        image = ImageHelper.pil_read_image(image_path)
        if self.configer.is_empty('test', 'test_input_size'):
            in_width, in_height = image.size
        else:
            in_width, in_height = self.configer.get('test', 'test_input_size')

        input_list = list()
        for scale in self.configer.get('test', 'scale_search'):
            image = Scale(size=(int(in_width * scale), int(in_height * scale)))(image)
            input_list.append(image)

        return input_list

    def make_mirror_input(self, image_path):
        image = ImageHelper.pil_read_image(image_path)
        if self.configer.is_empty('test', 'test_input_size'):
            in_width, in_height = image.size
        else:
            in_width, in_height = self.configer.get('test', 'test_input_size')

        image = Scale(size=(in_width, in_height))(image)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def tensor2bgr(self, tensor):
        assert len(tensor.size()) == 3

        ori_img = DeNormalize(mean=self.configer.get('trans_params', 'mean'),
                              std=self.configer.get('trans_params', 'std'))(tensor.cpu())
        ori_img = ori_img.numpy().transpose(1, 2, 0).astype(np.uint8)

        if self.configer.get('data', 'input_mode') == 'BGR':
            return ori_img
        else:
            image_bgr = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            return image_bgr
