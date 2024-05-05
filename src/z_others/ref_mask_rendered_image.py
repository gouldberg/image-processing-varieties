
import os
import glob
import json

import numpy as np

import math

import cv2
import PIL.Image

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------------------
# load rendered image and 2d coordinate 
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/blender/scripts/box_rotation_and_render/output'

# ----------
# coordinate file
coord_path = os.path.join(base_path, 'coords2d_dict.json')

with open(coord_path, 'r') as f:
    coord = json.load(f)


# ----------
image_path_list = sorted(glob.glob(os.path.join(base_path, '*.png')))


# idx = 25
# idx = 12

for idx in range(0, 5):
    img = cv2.imread(image_path_list[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    img_fname = os.path.basename(image_path_list[idx])
    img_copy = img.copy()
    # --------------------------------------------------------------------------------------------------
    # dot coordinate point
    # --------------------------------------------------------------------------------------------------
    # height and width
    resolution = img.shape[:2]
    # (x, y, z) in view plane --> cv2 (x, height - y, z)
    coord_obj = coord[img_fname]
    # print(coord_obj)
    for i in range(len(coord_obj)):
        center_coord = (int(coord_obj[i][0] * resolution[0]), int(resolution[1] - coord_obj[i][1] * resolution[1]))
        # center_coord = (int(coord_obj[i][1] * resolution[1]), int(resolution[0] - coord_obj[i][0] * resolution[0]))
        print(center_coord)
        img_copy = cv2.circle(
            img=img_copy,
            center=center_coord,
            radius=10,
            color=(255, 0, 0),
            thickness=3,
            lineType=cv2.LINE_4)
        # if i >= 1:
        #     img_to_show = np.hstack([img_to_show, img_copy])
    # PIL.Image.fromarray(img_to_show.astype('uint8')).show()
    PIL.Image.fromarray(img_copy.astype('uint8')).show()


# ----------
# idx = 0
# img = cv2.imread(image_path_list[idx])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.shape)

# img_fname = os.path.basename(image_path_list[idx])

# img_copy = img.copy()

# img_copy = cv2.circle(
#     img=img_copy,
#     center=(500,400),
#     radius=10,
#     color=(255, 0, 0),
#     thickness=3,
#     lineType=cv2.LINE_4)

# PIL.Image.fromarray(img_copy.astype('uint8')).show()



# --------------------------------------------------------------------------------------------------
# convert to convex hull and mask
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/blender/data/uv_sample/rendered_output'

# ----------
# coordinate file
coord_path = os.path.join(base_path, 'coords2d_dict.json')

with open(coord_path, 'r') as f:
    coord = json.load(f)


# ----------
image_path_list = sorted(glob.glob(os.path.join(base_path, '*.png')))


# idx = 25
idx = 12


# ----------
img = cv2.imread(image_path_list[idx])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_copy = img.copy()

# print(img.shape)
img_fname = os.path.basename(image_path_list[idx])
resolution = img.shape[:2]

coord_obj = coord[img_fname]

coord2 = []

for i in range(len(coord_obj)):
    center_coord = (int(coord_obj[i][0] * resolution[1]), int(resolution[0] - coord_obj[i][1] * resolution[1]))
    coord2.append(center_coord)

contour = np.array(coord2)[np.newaxis, :].astype('int32')
hull = cv2.convexHull(contour)

mask = np.zeros(img.shape, dtype=np.uint8)
mask = cv2.fillPoly(mask, pts=[hull], color=(255,255,255))
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

for i in range(len(hull)):
    img_copy = cv2.circle(
        img=img_copy,
        center=(hull[i][0][0], hull[i][0][1]),
        radius=10,
        color=(255, 0, 0),
        thickness=3,
        lineType=cv2.LINE_4)

img_to_show = np.hstack([img, img_copy, mask])
PIL.Image.fromarray(img_to_show.astype('uint8')).show()



