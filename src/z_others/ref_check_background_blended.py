
import os
import glob

import numpy as np
import numpy.matlib

import math
import random

import cv2
import PIL.Image


base_path = '/home/kswada/kw/image_processing'


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

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


# ----------
image_dir = os.path.join(base_path, '00_sample_images/prize/brassband_trading_badge')
img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

image_dir = os.path.join(base_path, '00_sample_images/cushion')
img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

image_dir = os.path.join(base_path, '00_sample_images/cushion/pacman')
img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))


# --------------------------------------------------------------------------------------------------
# background images to be blended
# --------------------------------------------------------------------------------------------------

image_dir_bg = os.path.join(base_path, '00_sample_images/background')

img_file_list_bg = sorted(glob.glob(os.path.join(image_dir_bg, '*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir_bg, '*.png')))

print(img_file_list_bg)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# check blending - 0:  original image + background image
# --------------------------------------------------------------------------------------------------

i = 0

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

# ----------
img_to_show0 = np.hstack([
    img_rgb, img_rgb,
    img_rgb, img_rgb])
img_to_show1 = img_to_show0.copy()
img_to_show2 = img_to_show0.copy()
img_to_show = np.vstack([img_to_show0, img_to_show1, img_to_show2])

# ----------
PIL.Image.fromarray(img_to_show.astype('uint8')).show()

# ----------
h = img_to_show.shape[0]
w = img_to_show.shape[1]

for bg_idx in range(len(img_file_list_bg)):
    img_file_bg = img_file_list_bg[bg_idx]
    img_bg = cv2.imread(img_file_bg)
    img_bg = cv2.resize(img_bg, (w, h))
    img_bg_rgb = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
    img_blended = (img_to_show * 0.5 + img_bg_rgb * 0.5).astype('uint8')
    PIL.Image.fromarray(img_blended.astype('uint8')).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# check blending - 1:  original image + original image
# --------------------------------------------------------------------------------------------------

image_dir_1 = os.path.join(base_path, '00_sample_images/cushion/pacman')
img_file_list_1 = sorted(glob.glob(os.path.join(image_dir_1, '*.jpg')))

# image_dir_2 = os.path.join(base_path, '00_sample_images/cushion')
# img_file_list_2 = sorted(glob.glob(os.path.join(image_dir_2, '*.jpg')))

image_dir_2 = os.path.join(base_path, '00_sample_images/cushion/pacman')
img_file_list_2 = sorted(glob.glob(os.path.join(image_dir_2, '*.jpg')))

idx1 = 2
idx2 = 3


# ----------
img_file = img_file_list_1[idx1]
img = cv2.imread(img_file)
height = img.shape[0]
width = img.shape[1]
width_resize = 640
scale = width_resize / width
height_resize = int(height * scale)
width_resize = int(width * scale)
img = cv2.resize(img, (width_resize, height_resize))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------
img_to_show0 = np.hstack([
    img_rgb, img_rgb,
    img_rgb, img_rgb])
img_to_show1 = img_to_show0.copy()
img_to_show2 = img_to_show0.copy()
img_to_show_front = np.vstack([img_to_show0, img_to_show1, img_to_show2])


# ----------
img_file = img_file_list_2[idx2]
img = cv2.imread(img_file)
height = img.shape[0]
width = img.shape[1]
width_resize = 640
scale = width_resize / width
height_resize = int(height * scale)
width_resize = int(width * scale)
img = cv2.resize(img, (width_resize, height_resize))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------
img_to_show0 = np.hstack([
    img_rgb, img_rgb,
    img_rgb, img_rgb])
img_to_show1 = img_to_show0.copy()
img_to_show2 = img_to_show0.copy()
img_to_show_bg = np.vstack([img_to_show0, img_to_show1, img_to_show2])


h = img_to_show_front.shape[0]
w = img_to_show_front.shape[1]
img_to_show_bg = cv2.resize(img_to_show_bg, (w, h))


# ----------
img_blended = (img_to_show_front * 0.5 + img_to_show_bg * 0.5).astype('uint8')

PIL.Image.fromarray(img_to_show_front.astype('uint8')).show()
PIL.Image.fromarray(img_to_show_bg.astype('uint8')).show()
PIL.Image.fromarray(img_blended.astype('uint8')).show()
