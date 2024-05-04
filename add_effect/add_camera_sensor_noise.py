
import os
import glob

import numpy as np
import math
import random

import cv2
import PIL.Image


# ----------
# reference:
# https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py


####################################################################################################
# --------------------------------------------------------------------------------------------------
# add poisson noise to image to simulate camera sensor noise
# --------------------------------------------------------------------------------------------------

def get_random_state() -> np.random.RandomState:
    return np.random.RandomState(random.randint(0, (1 << 32) - 1))


def normal(loc, scale, size, random_state):
    if random_state is None:
        random_state = get_random_state()
    return random_state.normal(loc, scale, size)


def poisson(lam, size, random_state):
    if random_state is None:
        random_state = get_random_state()
    return random_state.poisson(lam, size)


def iso_noise(image, color_shift=0.05, intensity=0.5, random_state=None, **kwargs):
    """
    Apply poisson noise to image to simulate camera sensor noise.
    Args:
        image (numpy.ndarray): Input image, currently, only RGB, uint8 images are supported.
        color_shift (float):
        intensity (float): Multiplication factor for noise values. Values of ~0.5 are produce noticeable,
                   yet acceptable level of noise.
        random_state:
        **kwargs:
    Returns:
        numpy.ndarray: Noised image
    """
    # if image.dtype != np.uint8:
    #     raise TypeError("Image must have uint8 channel type")
    # if not is_rgb_image(image):
    #     raise TypeError("Image must be RGB")
    # ----------
    one_over_255 = float(1.0 / 255.0)
    image = np.multiply(image, one_over_255, dtype=np.float32)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)
    # ----------
    luminance_noise = poisson(stddev[1] * intensity * 255, size=hls.shape[:2], random_state=random_state)
    color_noise = normal(0, color_shift * 360 * intensity, size=hls.shape[:2], random_state=random_state)
    # ----------
    hue = hls[..., 0]
    hue += color_noise
    hue[hue < 0] += 360
    hue[hue > 360] -= 360
    # ----------
    luminance = hls[..., 1]
    luminance += (luminance_noise / 255) * (1.0 - luminance)
    # ----------
    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * 255
    return image.astype(np.uint8)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing/data_augmentation'

image_dir = os.path.join(base_path, 'sample_images')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')


# ----------
index = 0
img = cv2.imread(image_path_list[index])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------------------
# add poisson noise
# --------------------------------------------------------------------------------------------------

color_shift = 0.05
intensity = 0.5

img_noise = iso_noise(image=img, color_shirt=color_shift, intensity=intensity)

# PIL.Image.fromarray(img).show()
PIL.Image.fromarray(img_noise).show()




