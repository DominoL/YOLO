import types

import cv2
import numpy as np
import torch
from numpy import random
from PIL import Image
from torchvision import transforms


def rand_uniform(a=0, b=1):
    """
    均匀分布取随机数
    """
    return a + np.random.rand() * (b - a)

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, box=None, input_shape=None):
        for t in self.transforms:
            img, box, input_shape = t(img, box, input_shape)
        return img, box, input_shape

class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, box=None, input_shape=None):
        return self.lambd(img, box, input_shape)

class RandomFlip(object):
    """
    翻转图像
    """
    def __call__(self, image, box, input_shape):
        h, w = input_shape
        flip = rand_uniform()<.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0, 2]] = w - box[:, [2, 0]]

        return image, box, input_shape

class RandomAugmentHSV(object):
    """
    对图像进行色域变换
    """
    def __init__(self, hue=.1, sat=0.7, val=0.4):
        self.hue = hue
        self.sat = sat
        self.val = val

    def __call__(self, image, box=None, input_shape=None):
        image = np.array(image, np.uint8)
        #---------------------------------#
        #   随机得到色域变换的参数
        #---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [self.hue, self.sat, self.val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        dtype = image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        return image, box, input_shape



