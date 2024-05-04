
import os
import glob

import cv2
import numpy as np

import PIL.Image, PIL.ImageDraw, PIL.ImageFilter


# ----------
# reference:
# https://note.nkmk.me/python-pillow-composite/


base_path = '/home/kswada/kw/image_processing'
image_dir = os.path.join(base_path, '00_sample_images')


####################################################################################################
# --------------------------------------------------------------------------------------------------
# im1 0.5 + im2 0.5:  alpha blend by PIL.Image.Composite
# --------------------------------------------------------------------------------------------------

img_fname1 = 'beach.png'
img_fname2 = 'bedroom.jpg'


im1 = PIL.Image.open(os.path.join(image_dir, img_fname1))
im2 = PIL.Image.open(os.path.join(image_dir, img_fname2)).resize(im1.size)


# ----------
# “L”:  8 bit gray scale
mask02 = PIL.Image.new("L", im1.size, int(256*0.2))
mask05 = PIL.Image.new("L", im1.size, int(256*0.5))
mask08 = PIL.Image.new("L", im1.size, int(256*0.8))

img_blend02 = PIL.Image.composite(im1, im2, mask02)
img_blend05 = PIL.Image.composite(im1, im2, mask05)
img_blend08 = PIL.Image.composite(im1, im2, mask08)

img_to_show0 = np.hstack([np.array(im1), np.array(im2), np.array(im2)])
img_to_show1 = np.hstack([np.array(img_blend02), np.array(img_blend05), np.array(img_blend08)])
img_to_show = np.vstack([img_to_show0, img_to_show1])


# ----------
PIL.Image.fromarray(img_to_show.astype('uint8')).show()

PIL.Image.fromarray(np.array(mask)).show()



# --------------------------------------------------------------------------------------------------
# im1 + im2 gradiation
# --------------------------------------------------------------------------------------------------

im1 = PIL.Image.open(os.path.join(image_dir, img_fname1))
im2 = PIL.Image.open(os.path.join(image_dir, img_fname2)).resize(im1.size)

mask = PIL.Image.new("L", (600, 256))

draw = PIL.ImageDraw.Draw(mask)

for i in range(256):
    draw.line([(0, i), (600, i)], fill=i, width=1)


mask = mask.resize(im1.size)


# ----------
img_blend = PIL.Image.composite(im1, im2, mask)

img_to_show = np.hstack([np.array(im1), np.array(im2), np.array(img_blend)])


# ----------
PIL.Image.fromarray(img_to_show.astype('uint8')).show()

PIL.Image.fromarray(np.array(mask)).show()



# --------------------------------------------------------------------------------------------------
# im1 + im2 crop (masked)
# --------------------------------------------------------------------------------------------------

im1 = PIL.Image.open(os.path.join(image_dir, img_fname1))
im2 = PIL.Image.open(os.path.join(image_dir, img_fname2)).resize(im1.size)

mask = PIL.Image.new("L", im1.size, 0)

draw = PIL.ImageDraw.Draw(mask)

draw.ellipse((140, 50, 260, 170), fill=255)

# additionally mask:  blurred
mask = mask.filter(PIL.ImageFilter.GaussianBlur(10))

# ----------
img_blend = PIL.Image.composite(im1, im2, mask)

img_to_show = np.hstack([np.array(im1), np.array(im2), np.array(img_blend)])


# ----------
PIL.Image.fromarray(img_to_show.astype('uint8')).show()

PIL.Image.fromarray(np.array(mask)).show()


# --------------------------------------------------------------------------------------------------
# im1 + im2 (already masked)
# --------------------------------------------------------------------------------------------------

img_fname3 = 'masked_horse.png'

im1 = PIL.Image.open(os.path.join(image_dir, img_fname1))
im3 = PIL.Image.open(os.path.join(image_dir, img_fname3)).resize(im1.size)


mask = im3.convert('L').resize(im1.size)

# invert nega - posi
# mask = PIL.ImageChops.invert(mask)


# ----------
img_blend = PIL.Image.composite(im1, im2, mask)

img_to_show = np.hstack([np.array(im1), np.array(im2), np.array(img_blend)])


# ----------
PIL.Image.fromarray(img_to_show.astype('uint8')).show()

PIL.Image.fromarray(np.array(mask)).show()



####################################################################################################
# --------------------------------------------------------------------------------------------------
# im1 0.5 + im2 0.5:  alpha blend by cv2.addweighted()
# --------------------------------------------------------------------------------------------------

img_fname1 = 'beach.png'
img_fname2 = 'bedroom.jpg'


im1 = PIL.Image.open(os.path.join(image_dir, img_fname1))
im2 = PIL.Image.open(os.path.join(image_dir, img_fname2)).resize(im1.size)

im1 = np.array(im1)
im2 = np.array(im2)


# ----------
gamma = 0.

alpha = 0.2
img_blended02 = cv2.addWeighted(im1, alpha, im2, 1 - alpha, gamma)

alpha = 0.5
img_blended05 = cv2.addWeighted(im1, alpha, im2, 1 - alpha, gamma)

alpha = 0.8
img_blended08 = cv2.addWeighted(im1, alpha, im2, 1 - alpha, gamma)

img_to_show0 = np.hstack([im1, im2, im2])
img_to_show1 = np.hstack([img_blend02, img_blend05, img_blend08])
img_to_show = np.vstack([img_to_show0, img_to_show1])


# ----------
PIL.Image.fromarray(img_to_show.astype('uint8')).show()


# --------------------------------------------------------------------------------------------------
# im1 + im2 gradiation (by linspace)
# --------------------------------------------------------------------------------------------------

img_fname1 = 'beach.png'
img_fname2 = 'bedroom.jpg'


im1 = PIL.Image.open(os.path.join(image_dir, img_fname1))
im2 = PIL.Image.open(os.path.join(image_dir, img_fname2)).resize(im1.size)

im1 = np.array(im1)
im2 = np.array(im2)


# ----------
h = im1.shape[0]
w = im1.shape[1]

# horizontally and vertically
alpha_w = np.linspace(0, 1, w).reshape(1, -1, 1)
alpha_h = np.linspace(0, 1, h).reshape(-1, 1, 1)

img_blended_w = im1 * alpha_w + im2 * (1 - alpha_w)
img_blended_h = im1 * alpha_h + im2 * (1 - alpha_h)


# ----------
img_to_show = np.hstack([im1, im2, img_blended_w, img_blended_h])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# image composition:  im1 (to be masked: foreground) + im2 (background)
# --------------------------------------------------------------------------------------------------

img_fname1 = 'pokemon.jpg'
img_fname2 = 'bedroom.jpg'

im1 = PIL.Image.open(os.path.join(image_dir, img_fname1))
im2 = PIL.Image.open(os.path.join(image_dir, img_fname2))

fg_img = np.array(im1)
bg_img = np.array(im2)

# convert foreground to hsv
hsv = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)

# binarize
bin_img = cv2.inRange(hsv, (0, 10, 0), (255, 255, 255))

# contour
contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# countour with max area
contour = max(contours, key=lambda x: cv2.contourArea(x))

# generate mask image
mask = np.zeros_like(bin_img)
cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)


# composite
x, y = 10, 10

w = min(fg_img.shape[1], bg_img.shape[1] - x)
h = min(fg_img.shape[0], bg_img.shape[0] - y)

fg_roi = fg_img[:h, :w]
bg_roi = bg_img[y : y + h, x : x + w]

bg_roi[:] = np.where(mask[:h, :w, np.newaxis] == 0, bg_roi, fg_roi)

# ----------
PIL.Image.fromarray(fg_roi.astype('uint8')).show()
PIL.Image.fromarray(bg_roi.astype('uint8')).show()

PIL.Image.fromarray(bg_img.astype('uint8')).show()

PIL.Image.fromarray(mask.astype('uint8')).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# image composition:  im1 (to be masked: foreground + green background) + im2 (background)
# Chromekey Compositing (green background)
# --------------------------------------------------------------------------------------------------

img_fname1 = 'mask_image_for_chromekey.jpg'
img_fname2 = 'bedroom.jpg'

# this should be cv2
im1 = cv2.imread(os.path.join(image_dir, img_fname1))
im2 = cv2.imread(os.path.join(image_dir, img_fname2))

fg_img = np.array(im1)
bg_img = np.array(im2)

# convert foreground to hsv
hsv = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)

# binarize:  HERE is for CHROMEKEY (greend background)
bin_img = ~cv2.inRange(hsv, (62, 100, 0), (79, 255, 255))

# contour
contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# countour with max area
contour = max(contours, key=lambda x: cv2.contourArea(x))

# generate mask image
mask = np.zeros_like(bin_img)
cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)


# composite
x, y = 30, 10

w = min(fg_img.shape[1], bg_img.shape[1] - x)
h = min(fg_img.shape[0], bg_img.shape[0] - y)

fg_roi = fg_img[:h, :w]
bg_roi = bg_img[y : y + h, x : x + w]

dst = np.where(mask[:h, :w, np.newaxis] == 0, bg_roi, fg_roi)


# ----------
PIL.Image.fromarray(cv2.cvtColor(fg_img.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(bg_img.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(dst.astype('uint8'), cv2.COLOR_BGR2RGB)).show()

