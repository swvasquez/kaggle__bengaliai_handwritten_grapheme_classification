import json

import cv2
import lz4.frame
import numpy as np


def reshape(config, image1d):
    height = int(config['init_height'])
    width = int(config['init_width'])
    image1d.shape = (height, width)

    return image1d


def pad(config, image):
    padding = int(config['padding'])
    height, width = image.shape
    max_length = max(height, width)

    ypad = (
        padding + (max_length - height) // 2,
        padding + (max_length - height) - (max_length - height) // 2
    )

    xpad = (
        padding + (max_length - width) // 2,
        padding + (max_length - width) - (max_length - width) // 2
    )
    padded = np.pad(
        image,
        (ypad, xpad),
        'constant',
        constant_values=0
    )

    return padded


def resize(config, image):
    size = int(config['final_size'])
    resized = cv2.resize(image, (size, size))

    return resized


def as_json(config, image):
    as_array = image.tolist()
    as_json = json.dumps(as_array)

    return as_json


def compress(config, image):
    compression = int(config['compression_lvl'])
    image = lz4.frame.compress(image.encode(), compression_level=compression)

    return image

