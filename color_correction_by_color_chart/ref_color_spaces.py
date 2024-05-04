
import os
import glob
import shutil

import numpy as np
import cv2
import PIL.Image

import matplotlib.pyplot as plt

import pandas as pd

pd.set_option('display.max_rows', 80)
pd.set_option('display.max_columns', 20)


# --------------------------------------------------------------------------------------------------
# LAB space
# --------------------------------------------------------------------------------------------------

labl_list = [20, 70, 128, 180, 225, 255]

for labl in labl_list:
    img_lab = []
    for laba in np.arange(0, 255, 0.5):
        val = []
        for labb in np.arange(0, 255, 0.5):
            val.append([labl, laba, labb])
        img_lab.append(val)
    print(len(img_lab))
    print(img_lab[0])
    img_lab = np.array(img_lab, np.uint8)
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    PIL.Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# HSV space
# --------------------------------------------------------------------------------------------------

hsvv_list = [20, 70, 128, 180, 225, 255]

for hsvv in hsvv_list:
    img_hsv = []
    for hsvh in np.arange(-360, 360, 0.5) % 180:
        val = []
        for hsvs in np.arange(0, 255, 0.5):
            val.append([hsvh, hsvs, hsvv])
        img_hsv.append(val)
    img_hsv = np.array(img_hsv, np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    PIL.Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# RGB space
# --------------------------------------------------------------------------------------------------

b_list = [20, 70, 128, 180, 225, 255]

for b in b_list:
    img_rgb = []
    for r in np.arange(0, 255, 0.5):
        val = []
        for g in np.arange(0, 255, 0.5):
            val.append([r, g, b])
        img_rgb.append(val)
    img_rgb = np.array(img_rgb, np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    PIL.Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).show()

