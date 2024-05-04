
import os
import glob

import numpy as np

import math

import cv2
import PIL.Image

import matplotlib.pyplot as plt

import copy


####################################################################################################
# --------------------------------------------------------------------------------------------------
# image blending with plastic cover background
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir_plastic = os.path.join(base_path, '00_sample_images/plastic_cover_texture')

image_dir_prize = os.path.join(base_path, '00_sample_images/prize')

image_path_list_plastic = sorted(glob.glob(os.path.join(image_dir_plastic, '*.jpg')) + glob.glob(os.path.join(image_dir_plastic, '*.png')))
image_path_list_prize = sorted(glob.glob(os.path.join(image_dir_prize, '*.jpg')) + glob.glob(os.path.join(image_dir_prize, '*.png')))


# ----------
# save dir
save_dir = os.path.join(base_path, '00_sample_images/prize/prize_with_plastic_cover_bg_blended')

# blend weight 
weight = 0.5

for i in range(len(image_path_list_prize)):
    for j in range(len(image_path_list_plastic)):
        # ----------
        # load prize
        img_front = cv2.imread(image_path_list_prize[i])
        img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
        img_shape = img_front.shape
        # ----------
        # load back
        img_back = cv2.imread(image_path_list_plastic[j])
        img_back = cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB)
        img_back = cv2.resize(img_back, (int(img_shape[1]), int(img_shape[0])))
        # ----------
        # blend
        img_blended = (img_front * (1 - weight) + img_back * weight).astype('uint8')
        # ----------
        img_to_show = np.hstack([img_front, img_back, img_blended])
        # PIL.Image.fromarray(img_to_show).show()
        img_to_show2 = cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)
        save_fname = os.path.basename(image_path_list_prize[i]).split('.')[0] + '_' + os.path.basename(image_path_list_plastic[j]).split('.')[0] + '.png'
        save_path = os.path.join(save_dir, save_fname)
        cv2.imwrite(save_path, img_to_show2)

