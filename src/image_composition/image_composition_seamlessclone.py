
import os
import glob

import cv2
import numpy as np

import PIL.Image, PIL.ImageDraw, PIL.ImageFilter

import math


# ----------
# reference:
# http://amroamroamro.github.io/mexopencv/opencv/cloning_demo.html


base_path = '/home/kswada/kw/image_processing'
image_dir = os.path.join(base_path, '00_sample_images/seamless_cloning')


####################################################################################################
# --------------------------------------------------------------------------------------------------
# cv2.seamlessclone:  Normal Cloning
# --------------------------------------------------------------------------------------------------

dst_fname = 'test005_dst.png'
src_fname = 'test005_src.png'
mask_fname = 'test005_mask.png'

dst = cv2.imread(os.path.join(image_dir, dst_fname))
src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))

print(f'{dst.shape} - {src.shape}')

p = (400, 100)
result = cv2.seamlessClone(src, dst, mask, p, cv2.NORMAL_CLONE)


# ----------
PIL.Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# cv2.seamlessclone:  Mixed Cloning
# --------------------------------------------------------------------------------------------------

dst_fname = 'test001_dst.png'
src_fname = 'test001_src.png'
mask_fname = 'test001_mask.png'

dst = cv2.imread(os.path.join(image_dir, dst_fname))
src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))

print(f'{dst.shape} - {src.shape}')

p = (int(dst.shape[0]/2), int(dst.shape[1]/2))
result = cv2.seamlessClone(src, dst, mask, p, cv2.MIXED_CLONE)


# ----------
PIL.Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# cv2.seamlessclone:  Monochrome Transfer  (texture is transfered)
# --------------------------------------------------------------------------------------------------

dst_fname = 'test004_dst.png'
src_fname = 'test004_src.png'
mask_fname = 'test004_mask.png'

dst = cv2.imread(os.path.join(image_dir, dst_fname))
src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))

print(f'{dst.shape} - {src.shape}')

p = (int(dst.shape[0]/2), int(dst.shape[1]/2))
result = cv2.seamlessClone(src, dst, mask, p, cv2.MONOCHROME_TRANSFER)


# ----------
PIL.Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# cv2.seamlessclone
# --------------------------------------------------------------------------------------------------

img_fname1 = 'pokemon.jpg'
img_fname2 = 'bedroom.jpg'

src = cv2.imread(os.path.join(image_dir, img_fname1))
dst = cv2.imread(os.path.join(image_dir, img_fname2))

src = cv2.resize(src, (200, 200))


# ----------
# generate mask
# convert foreground to hsv
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# binarize
bin_img = cv2.inRange(hsv, (0, 10, 0), (255, 255, 255))

# contour
contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# countour with max area
contour = max(contours, key=lambda x: cv2.contourArea(x))

# generate mask image
mask = np.zeros_like(bin_img)
cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)


PIL.Image.fromarray(cv2.cvtColor(src.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(dst.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
point = (100, 100)
img_blended = cv2.seamlessClone(src, dst, mask, point, cv2.MIXED_CLONE)
img_blended = cv2.seamlessClone(src, dst, mask, point, cv2.NORMAL_CLONE)
img_blended = cv2.seamlessClone(src, dst, mask, point, cv2.MONOCHROME_TRANSFER)

PIL.Image.fromarray(cv2.cvtColor(img_blended.astype('uint8'), cv2.COLOR_BGR2RGB)).show()



