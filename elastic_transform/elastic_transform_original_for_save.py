
import os
import glob

import numpy as np
import math
import random

import cv2
import PIL.Image

# from scipy.ndimage import gaussian_filter
# from scipy.ndimage import map_coordinates
# from scipy.ndimage.interpolation import map_coordinates
# from scipy.ndimage.filters import gaussian_filter


# ----------
# reference:
# https://github.com/albumentations-team/albumentations/blob/b773a1aa69f9c823c7f593205614d05d32c039cb/albumentations/augmentations/geometric/functional.py
# https://gist.github.com/ernestum/601cdf56d2b424757de5


# ####################################################################################################
# # --------------------------------------------------------------------------------------------------
# # elastic transform
# # --------------------------------------------------------------------------------------------------

# # from typing import List, Optional, Sequence, Tuple, Union, Callable, Any
# # from functools import wraps
# # # from typing_extensions import Concatenate, ParamSpec

# # ImageColorType = Union[float, Sequence[float]]
# # NumType = Union[int, float, np.ndarray]

# def get_random_state() -> np.random.RandomState:
#     return np.random.RandomState(random.randint(0, (1 << 32) - 1))

# # def rand(d0: NumType, d1: NumType, *more, random_state: Optional[np.random.RandomState] = None, **kwargs) -> Any:
# def rand(d0, d1, *more, random_state, **kwargs):
#     if random_state is None:
#         random_state = get_random_state()
#     return random_state.rand(d0, d1, *more, **kwargs)  # type: ignore

# # def get_num_channels(image: np.ndarray) -> int:
# #     return image.shape[2] if len(image.shape) == 3 else 1

# def uniform(low, high, size, random_state):
#     if random_state is None:
#         random_state = get_random_state()
#     return random_state.uniform(low, high, size)

# # def _maybe_process_in_chunks(process_fn, **kwargs):
# #     """
# #     Wrap OpenCV function to enable processing images with more than 4 channels.
# #     Limitations:
# #         This wrapper requires image to be the first argument and rest must be sent via named arguments.
# #     Args:
# #         process_fn: Transform function (e.g cv2.resize).
# #         kwargs: Additional parameters.
# #     Returns:
# #         numpy.ndarray: Transformed image.
# #     """
# #     @wraps(process_fn)
# #     def __process_fn(img: np.ndarray) -> np.ndarray:
# #         num_channels = get_num_channels(img)
# #         if num_channels > 4:
# #             chunks = []
# #             for index in range(0, num_channels, 4):
# #                 if num_channels - index == 2:
# #                     # Many OpenCV functions cannot work with 2-channel images
# #                     for i in range(2):
# #                         chunk = img[:, :, index + i : index + i + 1]
# #                         chunk = process_fn(chunk, **kwargs)
# #                         chunk = np.expand_dims(chunk, -1)
# #                         chunks.append(chunk)
# #                 else:
# #                     chunk = img[:, :, index : index + 4]
# #                     chunk = process_fn(chunk, **kwargs)
# #                     chunks.append(chunk)
# #             img = np.dstack(chunks)
# #         else:
# #             img = process_fn(img, **kwargs)
# #         return img
# #     return __process_fn


# # ----------
# def elastic_transform(img, alpha, sigma, alpha_affine,
#                       interpolation=None, border_mode=None,
#                       value=None, random_state=None, approximate=False, same_dxdy=False):
#     """Elastic deformation of images as described in [Simard2003]_ (with modifications).
#     Based on https://gist.github.com/ernestum/601cdf56d2b424757de5
#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#          Convolutional Neural Networks applied to Visual Document Analysis", in
#          Proc. of the International Conference on Document Analysis and
#          Recognition, 2003.
#     """
#     # ----------
#     if interpolation is None:
#         intepolation = cv2.INTER_LINEAR
#     if border_mode is None:
#         border_mode = cv2.BORDER_REFLECT_101
#     # ----------
#     height, width = img.shape[:2]
#     # ----------
#     # Random affine
#     center_square = np.array((height, width), dtype=np.float32) // 2
#     square_size = min((height, width)) // 3
#     alpha = float(alpha)
#     sigma = float(sigma)
#     alpha_affine = float(alpha_affine)
#     # ----------
#     pts1 = np.array(
#         [
#             center_square + square_size,
#             [center_square[0] + square_size, center_square[1] - square_size],
#             center_square - square_size,
#         ],
#         dtype=np.float32,
#     )
#     pts2 = pts1 + uniform(-alpha_affine, alpha_affine, size=pts1.shape, random_state=random_state).astype(
#         np.float32
#     )
#     matrix = cv2.getAffineTransform(pts1, pts2)
#     # ----------
#     # warp_fn = _maybe_process_in_chunks(
#     #     cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
#     # )
#     # img = warp_fn(img)
#     img = cv2.warpAffine(img, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value)
#     # ----------
#     if approximate:
#         # Approximate computation smooth displacement map with a large enough kernel.
#         # On large images (512+) this is approximately 2X times faster
#         dx = rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
#         cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
#         dx *= alpha
#         if same_dxdy:
#             # Speed up even more
#             dy = dx
#         else:
#             dy = rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
#             cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
#             dy *= alpha
#     else:
#         # dx = np.float32(
#         #     gaussian_filter((rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
#         # )
#         # ----------
#         # no need sci
#         dx = (cv2.GaussianBlur((rand(height, width, random_state=random_state) * 2 - 1),
#                                (0, 0), sigma) * alpha).astype(np.float32)
#         if same_dxdy:
#             # Speed up
#             dy = dx
#         else:
#             # dy = np.float32(
#             #     gaussian_filter((rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
#             # )
#             dy = (cv2.GaussianBlur((rand(height, width, random_state=random_state) * 2 - 1),
#                                 (0, 0), sigma) * alpha).astype(np.float32)
#     # ----------
#     x, y = np.meshgrid(np.arange(width), np.arange(height))
#     # ----------
#     map_x = np.float32(x + dx)
#     map_y = np.float32(y + dy)
#     # ----------
#     # remap_fn = _maybe_process_in_chunks(
#     #     cv2.remap, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value
#     # )
#     # return remap_fn(img)
#     # ----------
#     return cv2.remap(img, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value)


# --------------------------------------------------------------------------------------------------
# elastic transform:  reformat version
# --------------------------------------------------------------------------------------------------

def get_random_state():
    return np.random.RandomState(random.randint(0, (1 << 32) - 1))

def rand(d0, d1, *more, random_state, **kwargs):
    if random_state is None:
        random_state = get_random_state()
    return random_state.rand(d0, d1, *more, **kwargs)

def uniform(low, high, size, random_state):
    if random_state is None:
        random_state = get_random_state()
    return random_state.uniform(low, high, size)

# ----------
def elastic_transform(img, alpha, sigma, alpha_affine,
                      interpolation=None, border_mode=None,
                      value=None, random_state=None, approximate=False, same_dxdy=False):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    # ----------
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR
    if border_mode is None:
        border_mode = cv2.BORDER_REFLECT_101
    # ----------
    height, width = img.shape[:2]
    # ----------
    # Random affine
    center_square = np.array((height, width), dtype=np.float32) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)
    # ----------
    pts1 = np.array(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ],
        dtype=np.float32,
    )
    pts2 = pts1 + uniform(-alpha_affine, alpha_affine, size=pts1.shape, random_state=random_state).astype(
        np.float32
    )
    matrix = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value)
    # ----------
    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha
        if same_dxdy:
            # Speed up even more
            dy = dx
        else:
            dy = rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
            cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
            dy *= alpha
    else:
        dx = (cv2.GaussianBlur((rand(height, width, random_state=random_state) * 2 - 1),
                               (0, 0), sigma) * alpha).astype(np.float32)
        if same_dxdy:
            # Speed up
            dy = dx
        else:
            dy = (cv2.GaussianBlur((rand(height, width, random_state=random_state) * 2 - 1),
                                (0, 0), sigma) * alpha).astype(np.float32)
    # ----------
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    return cv2.remap(img, map1=np.float32(x + dx), map2=np.float32(y + dy),
                     interpolation=interpolation, borderMode=border_mode, borderValue=value)


# --------------------------------------------------------------------------------------------------
# elastic transform simple version
# --------------------------------------------------------------------------------------------------

from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

def elastic_transform_simple(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    # ----------
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    # ----------
    # x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    # ----------
    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


# --------------------------------------------------------------------------------------------------
# elastic transform simple version2
# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
# --------------------------------------------------------------------------------------------------

# from skimage.filters import gaussian

def elastic_transform_simple2(image, severity=1):
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]
    # ----------
    sigma = np.array(shape_size) * 0.01
    alpha = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12][severity - 1]
    max_dx = shape[0] * 0.005
    max_dy = shape[0] * 0.005
    # ----------
    # dx = (gaussian(np.random.uniform(-max_dx, max_dx, size=shape[:2]),
    #                sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)
    # dy = (gaussian(np.random.uniform(-max_dy, max_dy, size=shape[:2]),
    #                sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)
    # ----------
    # does the same as above
    dx = (cv2.GaussianBlur(np.random.uniform(-max_dx, max_dx, size=shape[:2]), 
                            (0, 0), sigmaX=sigma[1], sigmaY=sigma[0]) * alpha).astype(np.float32)
    dy = (cv2.GaussianBlur(np.random.uniform(-max_dy, max_dy, size=shape[:2]), 
                            (0, 0), sigmaX=sigma[1], sigmaY=sigma[0]) * alpha).astype(np.float32)
    # ----------
    if len(image.shape) < 3 or image.shape[2] < 3:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    else:
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    # ----------
    distorted_image = np.clip(
        map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255
    return distorted_image


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir = os.path.join(base_path, '00_sample_images')

image_dir = os.path.join(base_path, '00_sample_images')

# image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
# print(f'num of images: {len(image_path_list)}')


# ----------
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/blue/*')))
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/yellow/*')))
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/red/*')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.png')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, '*.png')))


# --------------------------------------------------------------------------------------------------
# elastic transform
# --------------------------------------------------------------------------------------------------

alpha1 = 100
sigma1 = 10
# alpha_affine = 100
alpha_affine = 0

sigma2 = 10
alpha2 = 1000

for i in range(len(img_file_list)):
    img_file = img_file_list[i]
    img = cv2.imread(img_file)
    height = img.shape[0]
    width = img.shape[1]
    width_resize = 640
    scale = width_resize / width
    height_resize = int(height * scale)
    width_resize = int(width * scale)
    img = cv2.resize(img, (width_resize, height_resize))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # --------------------------------------------------------------------------------------------------
    # elastic transform
    # --------------------------------------------------------------------------------------------------
    # ----------
    # approximate = False is better
    img_trans_rgb1 = elastic_transform(img_rgb, alpha=alpha1, sigma=sigma1, alpha_affine=alpha_affine)
    # img_trans_rgb1 = elastic_transform(img_rgb, alpha=alpha1, sigma=sigma1, alpha_affine=alpha_affine, approximate=True)
    # ----------
    img_trans_rgb2 = elastic_transform_simple(image=img_rgb, alpha=alpha2, sigma=sigma2)
    # ----------
    img_to_show = np.hstack([img_rgb, img_trans_rgb1, img_trans_rgb2])
    # ----------
    # for i in range(5):
    #     img_trans = elastic_transform_simple2(img, severity=i+1).astype('uint8')
    #     img_trans_rgb3 = cv2.cvtColor(img_trans, cv2.COLOR_BGR2RGB)
    #     img_to_show = np.hstack([img_to_show, img_trans_rgb3])
    # ----------
    PIL.Image.fromarray(img_to_show.astype('uint8')).show()

