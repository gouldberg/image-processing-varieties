
import os
import numpy as np

import PIL.Image

from skimage.data import astronaut

from collections.abc import Iterable
from scipy import ndimage as ndi

base_path = '/home/kswada/kw/image_processing'


# --------------------------------------------------------------------------------------------------
# scipy ndimage gaussian_filter
# --------------------------------------------------------------------------------------------------

a = np.zeros((3, 3))
a[1, 1] = 1


# ----------
print(a)

print(ndi.gaussian_filter(a, sigma=0.4))

# mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
# The mode parameter determines how the array borders are handled
# Default is 'nearest'.
print(ndi.gaussian_filter(a, sigma=0.4, mode='reflect'))


# --------------------------------------------------------------------------------------------------
# try astronaut image
# --------------------------------------------------------------------------------------------------

image = astronaut()

# (512, 512, 3)
print(image.shape)
PIL.Image.fromarray(image).show()

print(image.ndim)


# ----------
# channel_axis: If None, the image is assumed to be a grayscale (single channel) image.
# Otherwise, this parameter indicates which axis of the array corresponds to channels.
channel_axis = -1

# standard deviation of gaussian kernel
# ths standard deviations of the gaussian filter are given for each axis as a sequence,
# or as a single number, in which case it is equal for all axes.
sigma = 1

if channel_axis is not None:
    # do not filter across channels
    if not isinstance(sigma, Iterable):
        sigma = [sigma] * (image.ndim - 1)
    if len(sigma) == image.ndim - 1:
        sigma = list(sigma)
        sigma.insert(channel_axis % image.ndim, 0)

# [1, 1, 0]
print(sigma)

# sigma = [0, 0, 1]


# ----------
mode = 'nearest'

# value to fill past edges of input if mode is constant
# default is 0.0
cval = 0.

# truncate (float):
# truncate the filter at this many standard deviations.
truncate = 4.0


# ----------
# default is order = 0,  order = 1 is difference of gaussian
filtered_img = ndi.gaussian_filter(image, sigma, order=0, output=None, mode=mode, cval=cval, truncate=truncate)

img_to_show = np.hstack([image, filtered_img])
PIL.Image.fromarray(img_to_show).show()
