#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Generate the inputs.


from PIL import Image

from datasets.tools.transforms import Scale
from utils.helpers.image_helper import ImageHelper


class BlobHelper(object):
    def __init__(self, configer):
        self.configer = configer

    def make_input_list(self, image_path):
        image = ImageHelper.pil_open_rgb(image_path)
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
        image = ImageHelper.pil_open_rgb(image_path)
        if self.configer.is_empty('test', 'test_input_size'):
            in_width, in_height = image.size
        else:
            in_width, in_height = self.configer.get('test', 'test_input_size')

        image = Scale(size=(in_width, in_height))(image)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def make_crop_inputs(self):
        pass
