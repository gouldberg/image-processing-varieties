
import os
import glob

import cv2
import numpy as np

import PIL.Image, PIL.ImageDraw, PIL.ImageFilter

import math


base_path = '/home/kswada/kw/image_processing'


####################################################################################################
# --------------------------------------------------------------------------------------------------
# image composition 
# --------------------------------------------------------------------------------------------------

image_dir = os.path.join(base_path, '00_sample_images/seamless_cloning')

dst_fname = 'test002_dst.png'
src_fname = 'test002_src.png'
mask_fname = 'test002_mask.png'

dst = cv2.imread(os.path.join(image_dir, dst_fname))
src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))

dst_comp = dst.copy()


# ----------
# resize
src = cv2.resize(src, (dst.shape[1], dst.shape[0]))
mask = cv2.resize(mask, (dst.shape[1], dst.shape[0]))

print(f'{dst.shape} - {src.shape} - {mask.shape}')


# ----------
# crop source by mask
src_crop = cv2.bitwise_and(src, mask)


# ----------
# composition
dst_comp = dst.copy()
dst_comp[:] = np.where(mask == 0, dst, src_crop)

image_to_show = np.hstack([dst, src, src_crop, dst_comp])

PIL.Image.fromarray(cv2.cvtColor(image_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()



####################################################################################################
# --------------------------------------------------------------------------------------------------
# image composition considering coordinate
# --------------------------------------------------------------------------------------------------

image_dir = os.path.join(base_path, '00_sample_images/seamless_cloning')

dst_fname = 'test005_dst.png'
src_fname = 'test005_src.png'
mask_src_fname = 'test005_mask.png'

dst = cv2.imread(os.path.join(image_dir, dst_fname))
src = cv2.imread(os.path.join(image_dir, src_fname))
mask_src = cv2.imread(os.path.join(image_dir, mask_fname))
mask_dst = np.full(dst.shape, 255)

print(f'{dst.shape} - {mask_dst.shape} - {src.shape} - {mask_src.shape}')


# # ----------
# scale_dst = 0.5
# scale_src = 1.0

# dst = cv2.resize(dst, (int(dst.shape[1]*scale_dst), int(dst.shape[0]*scale_dst)))
# src = cv2.resize(src, (int(src.shape[1]*scale_src), int(src.shape[0]*scale_src)))
# mask_dst = cv2.resize(mask_dst, (int(mask_dst.shape[1]*scale_dst), int(mask_dst.shape[0]*scale_dst)))
# mask_src = cv2.resize(mask_src, (int(mask_src.shape[1]*scale_src), int(mask_src.shape[0]*scale_src)))

# print(f'{dst.shape} - {src.shape} - {mask_dst.shape} - {mask_src.shape}')


# ----------
x, y = 10, 10
w = min(src.shape[1], dst.shape[1] - x)
h = min(src.shape[0], dst.shape[0] - y)

src_roi = src[:h, :w]
dst_roi = dst[y : y + h, x : x + w]

print(f'{x} - {y} - {w} - {h}')
print(f'{dst_roi.shape} - {src_roi.shape} - {mask_src.shape} - {mask_src[:h, :w].shape}')

dst_comp = dst_roi.copy()
dst_comp[:] = np.where(mask_src[:h, :w] == 0, dst_roi, src_roi)


# ----------
image_to_show = np.hstack([dst_roi, src_roi, dst_comp])

PIL.Image.fromarray(cv2.cvtColor(image_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()

