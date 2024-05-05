
import os
import glob

import cv2
import numpy as np

import PIL.Image, PIL.ImageDraw, PIL.ImageFilter
import torchvision.transforms.functional as TF

import math

base_path = '/home/kswada/kw/image_processing'


####################################################################################################
# --------------------------------------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------------------------------------

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # !!! compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image:  dst, src(masked), mask
# --------------------------------------------------------------------------------------------------

image_dir = os.path.join(base_path, '00_sample_images/seamless_cloning')

dst_fname = 'test002_dst.png'
src_fname = 'test002_src.png'
mask_fname = 'test002_mask.png'

ret_img = cv2.imread(os.path.join(image_dir, dst_fname))
src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))


# ----------
# background resized
# scale_w, scale_h = 2.0, 1.5
scale_w, scale_h = 1.5, 1.5
ret_img = cv2.resize(ret_img, (int(ret_img.shape[1] * scale_w), int(ret_img.shape[0] * scale_h)))


# ----------
# src masked
src = cv2.bitwise_and(src, mask)


PIL.Image.fromarray(cv2.cvtColor(ret_img.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(src.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# transform src image
# --------------------------------------------------------------------------------------------------

angle = 30
# scale_w, scale_h = 1.5, 1.5
scale_w, scale_h = 2.5, 2.5

src_trans = cv2.resize(src, (int(src.shape[1] * scale_w), int(src.shape[0] * scale_h)))
src_trans = rotate(src_trans, angle)

mask_trans = cv2.resize(mask, (int(mask.shape[1] * scale_w), int(mask.shape[0] * scale_h)))
mask_trans = rotate(mask_trans, angle)

image_to_show = np.hstack([src_trans, mask_trans])
PIL.Image.fromarray(cv2.cvtColor(image_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# get bounding box coordinate of foreground
# --------------------------------------------------------------------------------------------------

gray = cv2.cvtColor(mask_trans, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

x, y, f_w, f_h = cv2.boundingRect(cnt)

# x = x - 2
# y = y - 2
# f_w = f_w + 4
# f_h = f_h + 4
# mask_trans_with_bbox = cv2.rectangle(mask_trans, (x, y), (x + f_w, y + f_h), (255, 255, 0), 2)
#
# PIL.Image.fromarray(cv2.cvtColor(mask_trans_with_bbox.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# composition with rood outside
# --------------------------------------------------------------------------------------------------

# the larger the coef is, the larger part is in inside
inside_coef = 0.2

for _ in range(10):

    comp_min_x = np.random.randint(0 - int(f_w * (1 - inside_coef)), ret_img.shape[1] - 1 - int(f_w * inside_coef))
    comp_min_y = np.random.randint(0 - int(f_h * (1 - inside_coef)), ret_img.shape[0] - 1 - int(f_h * inside_coef))
    comp_max_x = comp_min_x + f_w - 1
    comp_max_y = comp_min_y + f_h - 1

    # ----------
    comp_min_x2 = comp_min_x
    comp_max_x2 = comp_max_x
    comp_min_y2 = comp_min_y
    comp_max_y2 = comp_max_y
    flg_min_x = True
    flg_max_x = True
    flg_min_y = True
    flg_max_y = True

    if comp_min_x < 0:
        comp_min_x2 = 0
        flg_min_x = False
    if comp_max_x > ret_img.shape[1] - 1:
        comp_max_x2 = ret_img.shape[1] - 1
        flg_max_x = False
    if comp_min_y < 0:
        comp_min_y2 = 0
        flg_min_y = False
    if comp_max_y > ret_img.shape[0] - 1:
        comp_max_y2 = ret_img.shape[0] - 1
        flg_max_y = False

    if comp_max_x2 <= comp_min_x2 or comp_max_y2 <= comp_min_x2:
        continue

    f_w2 = comp_max_x2 - comp_min_x2 + 1
    f_h2 = comp_max_y2 - comp_min_y2 + 1

    x2 = x
    y2 = y

    if flg_min_x == False:
        x2 = x + f_w - f_w2
    if flg_min_y == False:
        y2 = y + f_h - f_h2

    print(f'flag                    min_x: {flg_min_x}  max_x: {flg_max_x}  min_y: {flg_min_y}  max_y: {flg_max_y}')
    print(f'ret img                 x:(0, {ret_img.shape[1] - 1})  y:(0, {ret_img.shape[0] - 1})  wh:({ret_img.shape[1]}, {ret_img.shape[0]})')
    print(f'foreground in orig      x:({x}, {x + f_w - 1})  y:({y}, {y + f_h - 1})  wh: ({f_w}, {f_h})')
    print(f'foreground in comp      x:({comp_min_x}, {comp_max_x})  y:({comp_min_y}, {comp_max_y})  wh: ({f_w}, {f_h})')
    print(f'foreground in comp rev  x:({comp_min_x2}, {comp_max_x2})  y:({comp_min_y2}, {comp_max_y2})  wh: ({f_w2}, {f_h2})')
    print(f'foreground in orig rev  x:({x2}, {x2 + f_w2 - 1})  y:({y2}, {y2 + f_h2 - 1})  wh: ({f_w2}, {f_h2})')

    # ----------
    fog_roi = src_trans[y2:y2 + f_h2, x2:x2 + f_w2]
    mask_roi = mask_trans[y2:y2 + f_h2, x2:x2 + f_w2]
    bg_roi = ret_img[comp_min_y2:comp_max_y2+1, comp_min_x2:comp_max_x2+1]

    print(f'fog_roi: {fog_roi.shape}  mask_roi: {mask_roi.shape}  bg_roi: {bg_roi.shape}')


    ret_img2 = ret_img.copy()
    ret_img2[comp_min_y2:comp_max_y2+1, comp_min_x2:comp_max_x2+1] = np.where(mask_roi == 0, bg_roi, fog_roi)

    # image_to_show = np.hstack([ret_img, ret_img2])
    # PIL.Image.fromarray(cv2.cvtColor(image_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()

    PIL.Image.fromarray(cv2.cvtColor(ret_img2.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
