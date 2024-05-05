
import os
import glob

import numpy as np
import math
import random

import cv2
import PIL.Image


# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing/data_augmentation'

image_dir = os.path.join(base_path, 'sample_images')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')


# ----------
# img_file = os.path.join(image_dir, 'flare_dog.png')
# img_file = os.path.join(image_dir, 'flare_dog_weak.png')
# img_file = os.path.join(image_dir, 'natural_flare.png')
# img_file = os.path.join(image_dir, 'gyakko_dog.png')
# img_file = os.path.join(image_dir, 'mountain_in_dark.png')
# img_file = os.path.join(image_dir, 'strong_light.jpg')
img_file = os.path.join(image_dir, 'yacho_kansatsu.jpg')
# img_file = os.path.join(image_dir, 'sun_reflection.jpg')


img = cv2.imread(img_file)

height = img.shape[0]
width = img.shape[1]
width_resize = 640

scale = width_resize / width

height_resize = int(height * scale)
width_resize = int(width * scale)

img = cv2.resize(img, (width_resize, height_resize))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------------------
# simple histogram equalization
# adaptive equalization by CLAHE
# --------------------------------------------------------------------------------------------------

# ----------
# by each R/G/B

clipLimit = 2.0
tile = 8
clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile, tile))

# simple equalization
img_eq_rgb = np.zeros(img.shape)
img_eq_rgb[:,:,0] = cv2.equalizeHist(img[:,:,0])
img_eq_rgb[:,:,1] = cv2.equalizeHist(img[:,:,1])
img_eq_rgb[:,:,2] = cv2.equalizeHist(img[:,:,2])

# clahe
img_clahe_rgb = np.zeros(img.shape)
img_clahe_rgb[:,:,0] = clahe.apply(img[:,:,0])
img_clahe_rgb[:,:,1] = clahe.apply(img[:,:,1])
img_clahe_rgb[:,:,2] = clahe.apply(img[:,:,2])

img_to_show = np.hstack([img, img_eq_rgb, img_clahe_rgb])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()



# ----------
# only Y in YUV

img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

clipLimit = 2.0
tile = 8
clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile, tile))

# simple equalization
img_eq_yuv = np.zeros(img_yuv.shape)
img_eq_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_eq_yuv[:,:,1] = img_yuv[:,:,1]
img_eq_yuv[:,:,2] = img_yuv[:,:,2]

# clahe
img_clahe_yuv = np.zeros(img.shape)
img_clahe_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
img_clahe_yuv[:,:,1] = img_yuv[:,:,1]
img_clahe_yuv[:,:,2] = img_yuv[:,:,2]


# ----------
img_eq = cv2.cvtColor(img_eq_yuv.astype('uint8'), cv2.COLOR_YUV2RGB)
img_clahe = cv2.cvtColor(img_clahe_yuv.astype('uint8'), cv2.COLOR_YUV2RGB)

img_to_show = np.hstack([img, img_eq, img_clahe])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()


# ----------
# by Y/U/V

img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

clipLimit = 2.0
tile = 8
clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile, tile))

# simple equalization
img_eq_yuv = np.zeros(img_yuv.shape)
img_eq_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_eq_yuv[:,:,1] = cv2.equalizeHist(img_yuv[:,:,1])
img_eq_yuv[:,:,2] = cv2.equalizeHist(img_yuv[:,:,2])

# clahe
img_clahe_yuv = np.zeros(img.shape)
img_clahe_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
img_clahe_yuv[:,:,1] = clahe.apply(img_yuv[:,:,1])
img_clahe_yuv[:,:,2] = clahe.apply(img_yuv[:,:,2])


img_eq = cv2.cvtColor(img_eq_yuv.astype('uint8'), cv2.COLOR_YUV2RGB)
img_clahe = cv2.cvtColor(img_clahe_yuv.astype('uint8'), cv2.COLOR_YUV2RGB)

img_to_show = np.hstack([img, img_eq, img_clahe])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()


# ----------
# by H/S/L

img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HSL)

clipLimit = 2.0
tile = 8
clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile, tile))

# simple equalization
img_eq_hsl = np.zeros(img_hsl.shape)
img_eq_hsl[:,:,0] = cv2.equalizeHist(img_hsl[:,:,0])
img_eq_hsl[:,:,1] = cv2.equalizeHist(img_hsl[:,:,1])
img_eq_hsl[:,:,2] = cv2.equalizeHist(img_hsl[:,:,2])

# clahe
img_clahe_hsl = np.zeros(img.shape)
img_clahe_hsl[:,:,0] = clahe.apply(img_hsl[:,:,0])
img_clahe_hsl[:,:,1] = clahe.apply(img_hsl[:,:,1])
img_clahe_hsl[:,:,2] = clahe.apply(img_hsl[:,:,2])


img_eq = cv2.cvtColor(img_eq_hsl.astype('uint8'), cv2.COLOR_HSL2RGB)
img_clahe = cv2.cvtColor(img_clahe_hsl.astype('uint8'), cv2.COLOR_HSL2RGB)

img_to_show = np.hstack([img, img_eq, img_clahe])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()
