#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


import numpy as np

from datasets.tools.transforms import Scale, ToTensor, Normalize
from utils.helpers.image_helper import ImageHelper
from utils.tools.logger import Logger as Log


class BlobHelper(object):

    def __init__(self, configer):
        self.configer = configer

    def get_img_blob(self, image_path=None):
        image = ImageHelper.pil_open_rgb(image_path)
        image = Scale(size=self.configer.get('data', 'input_size'))(image)
        image = ToTensor()(image)
        image = Normalize(mean=self.configer.get('trans_params', 'mean'),
                          std=self.configer.get('trans_params', 'std'))(image)
        image = image.unsqueeze(0)
        return image
