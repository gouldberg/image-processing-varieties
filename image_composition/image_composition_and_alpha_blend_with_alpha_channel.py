
import os
import cv2

import glob
import numpy as np

import PIL.Image, PIL.ImageDraw, PIL.ImageFilter


# --------------------------------------------------------------------------------------------------
# normal image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

img_dir = os.path.join(base_path, '00_sample_images/prize/brassband_trading_badge')

img_file_jpg = 'brassband_trading_badge_01.jpg'
img_file_png = 'brassband_trading_badge_01.png'


img_jpg1 = cv2.imread(os.path.join(img_dir, img_file_jpg))
img_png1 = cv2.imread(os.path.join(img_dir, img_file_png))

img_jpg2 = cv2.imread(os.path.join(img_dir, img_file_jpg), cv2.IMREAD_UNCHANGED)
# img_jpg2 = cv2.imread(os.path.join(img_dir, img_file_jpg), -1)
img_png2 = cv2.imread(os.path.join(img_dir, img_file_png), cv2.IMREAD_UNCHANGED)


print(img_jpg1.shape)
print(img_png1.shape)
print(img_jpg2.shape)
print(img_png2.shape)


# ----------
cv2.imwrite(os.path.join(img_dir, 'output.png'), img_jpg1, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
img_file_jpg = 'output.png'

img_png3 = cv2.imread(os.path.join(img_dir, 'output.png'), cv2.IMREAD_UNCHANGED)
print(img_png3.shape)


# --------------------------------------------------------------------------------------------------
# png image with alpha and alpha composite
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

img_file_src_png = os.path.join(base_path, '00_sample_images/alpha_channel/burdockStar.png')
img_file_dst_jpg = os.path.join(base_path, '00_sample_images/bedroom.jpg')

src = cv2.imread(img_file_src_png, -1)
dst = cv2.imread(img_file_dst_jpg)


# now src file have alpha channel
print(src.shape)
print(dst.shape)


# ----------
width, height = src.shape[:2]

# get alpha as mask
mask = src[:, :, 3]
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# for convenience, convert 0-255 to 0.0-1.0
mask2 = mask / 255
print(mask2.shape)

# remove alpha
src2 = src[:, :, :3]
print(src2.shape)

# alpha blend
dst_float = dst.astype(np.float64)
dst_float[0:height:, 0:width] *= 1 - mask2
dst_float[0:height:, 0:width] += src2 * mask2

dst_blended = dst_float.astype('uint8')

PIL.Image.fromarray(cv2.cvtColor(dst_blended.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
