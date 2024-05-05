
import os
import glob

import numpy as np
import math
import random

import cv2
import PIL.Image


####################################################################################################
# --------------------------------------------------------------------------------------------------
# reflection reduction by CLAHE to V
# --------------------------------------------------------------------------------------------------

def reflection_reduction_clahev(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv_image = cv2.merge([h, s, v])
    img_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return img_rgb


# --------------------------------------------------------------------------------------------------
# reflection reduction by thresholding and inpainting
# --------------------------------------------------------------------------------------------------

def reflection_reduction_inpaint(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]
    dst_TELEA = cv2.inpaint(img, thresh, 3, cv2.INPAINT_TELEA)
    return dst_TELEA



####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir = os.path.join(base_path, '00_sample_images')

# image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
# print(f'num of images: {len(image_path_list)}')


# ----------
# img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/blue/*')))
# img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/yellow/*')))
# img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/red/*')))

# img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.jpg'))) + \
#     sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.png')))

# img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
#     sorted(glob.glob(os.path.join(image_dir, '*.png')))


img_file_list = sorted(glob.glob(os.path.join(image_dir, 'reflection/*')))


# --------------------------------------------------------------------------------------------------
# reflection reduction
# --------------------------------------------------------------------------------------------------

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
    # reflection reduction
    # --------------------------------------------------------------------------------------------------
    img_reduced1 = reflection_reduction_clahev(img_rgb)
    img_reduced2 = reflection_reduction_inpaint(img_rgb)
    # ----------
    img_to_show = np.hstack([img_rgb, img_reduced1, img_reduced2])
    PIL.Image.fromarray(img_to_show.astype('uint8')).show()



