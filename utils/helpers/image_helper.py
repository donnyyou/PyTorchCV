#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Repackage some image operations.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from PIL import Image

from  utils.tools.logger import Logger as Log


class ImageHelper(object):

    @staticmethod
    def cv2_open_bgr(image_path):
        img_rgb = np.array(Image.open(image_path).convert('RGB'))
        return ImageHelper.rgb2bgr(img_rgb)

    @staticmethod
    def cv2_open_p(image_path):
        return np.array(Image.open(image_path).convert('P'))

    @staticmethod
    def pil_open_rgb(image_path):
        return Image.open(image_path).convert('RGB')

    @staticmethod
    def pil_open_p(image_path):
        return Image.open(image_path).convert('P')

    @staticmethod
    def rgb2bgr(img_rgb):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr

    @staticmethod
    def bgr2rgb(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    @staticmethod
    def imshow(win_name, img, time=0):
        if isinstance(img, Image.Image):
            img = ImageHelper.rgb2bgr(ImageHelper.img2np(img))

        cv2.imshow(win_name, img)
        cv2.waitKey(time)

    @staticmethod
    def draw_box(img, bbox, default_color=(255, 0, 255)):
        if isinstance(img, Image.Image):
            img = ImageHelper.rgb2bgr(ImageHelper.img2np(img))

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          color=default_color, thickness=3)
            return ImageHelper.np2img(ImageHelper.bgr2rgb(img))

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                      color=default_color, thickness=3)
        return img

    @staticmethod
    def np2img(arr):
        if len(arr.shape) == 2:
            mode = 'P'
        else:
            mode = 'RGB'

        return Image.fromarray(arr, mode=mode)

    @staticmethod
    def img2np(img):
        return np.array(img)

    @staticmethod
    def fig2np(fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    @staticmethod
    def resize(img, target_size, interpolation):
        assert isinstance(target_size, (list, tuple))

        if isinstance(img, Image.Image):
            return img.resize(target_size, interpolation)

        elif isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img, mode='P' if len(img.shape) == 2 else 'RGB')
            return np.array(pil_img.resize(target_size, interpolation))

        else:
            Log.error('Image type is invalid.')
            exit(1)

    @staticmethod
    def fig2img(fig):
        """
        @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
        @param fig a matplotlib figure
        @return a Python Imaging Library ( PIL ) image
        """
        # put the figure pixmap into a numpy array
        buf = ImageHelper.fig2data(fig)
        h, w, d = buf.shape
        return Image.frombytes("RGBA", (w, h), buf.tostring())

    @staticmethod
    def fig2data(fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        return buf.reshape(h, w, 4)

    @staticmethod
    def is_img(img_name):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]
        return any(img_name.endswith(extension) for extension in IMG_EXTENSIONS)


if __name__ == "__main__":
    target_size = (368, 368)
    image_path = '/home/donny/Projects/PytorchCV/val/samples/pose/coco/ski.jpg'
    pil_img = ImageHelper.pil_open_rgb(image_path)
    cv2_img = ImageHelper.cv2_open_bgr(image_path)

    pil_img = ImageHelper.resize(pil_img, target_size, interpolation=Image.CUBIC)
    cv2_img = ImageHelper.resize(cv2_img, target_size, interpolation=Image.CUBIC)
    cv2_img = ImageHelper.bgr2rgb(cv2_img)
    ImageHelper.imshow('main', np.array(pil_img) - cv2_img)
    ImageHelper.imshow('main', pil_img)
    ImageHelper.imshow('main', cv2_img)

    # resize_pil_img.show()
    print(np.unique(np.array(pil_img) - np.array(cv2_img)))