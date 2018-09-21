#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Generate the inputs.


import cv2
import numpy as np
import torch
from PIL import Image

from datasets.tools.transforms import DeNormalize, ToTensor, Normalize
from utils.helpers.image_helper import ImageHelper


class BlobHelper(object):
    def __init__(self, configer):
        self.configer = configer

    def make_input_batch(self, image_list, input_size=None, scale=1.0):
        input_list = list()
        for image in image_list:
            input_list.append(self.make_input(image, input_size=input_size, scale=scale))

        return torch.cat(input_list, 0)

    def make_input(self, image=None, input_size=None, scale=1.0):
        if input_size is None:
            in_width, in_height = ImageHelper.get_size(image)
        else:
            in_width, in_height = input_size

        image = ImageHelper.resize(image, (int(in_width * scale), int(in_height * scale)), interpolation=1)
        img_tensor = ToTensor()(image)
        img_tensor = Normalize(div_value=self.configer.get('normalize', 'div_value'),
                               mean=self.configer.get('normalize', 'mean'),
                               std=self.configer.get('normalize', 'std'))(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(torch.device('cpu' if self.configer.get('gpu') is None else 'cuda'))

        return img_tensor

    def tensor2bgr(self, tensor):
        assert len(tensor.size()) == 3

        ori_img = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                              mean=self.configer.get('normalize', 'mean'),
                              std=self.configer.get('normalize', 'std'))(tensor.cpu())
        ori_img = ori_img.numpy().transpose(1, 2, 0).astype(np.uint8)

        if self.configer.get('data', 'input_mode') == 'BGR':
            return ori_img
        else:
            image_bgr = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            return image_bgr
