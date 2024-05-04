
import os
import glob

import random
import math
import numpy as np

import cv2
import PIL.Image


# ----------
# reference:
# https://github.com/CrazyVertigo/imagecorruptions/blob/master/imagecorruptions/corruptions.py


####################################################################################################
# --------------------------------------------------------------------------------------------------
# add snow helpers
# --------------------------------------------------------------------------------------------------

# def clipped_zoom(img, zoom_factor):
#     # clipping along the width dimension:
#     ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
#     top0 = (img.shape[0] - ch0) // 2
#     # clipping along the height dimension:
#     ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
#     top1 = (img.shape[1] - ch1) // 2
#     img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
#                   (zoom_factor, zoom_factor, 1), order=1)
#     return img


def getOptimalKernelWidth1D(radius, sigma):
    return radius * 2 + 1


def gauss_function(x, mean, sigma):
    return (np.exp(- x**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)


def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k/Z


def shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[0]+dx, axis=0)
        shifted[dx:,:] = shifted[dx-1:dx,:]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=0)
        shifted[:dx,:] = shifted[dx:dx+1,:]
    else:
        shifted = image
    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[1]+dy, axis=1)
        shifted[:,dy:] = shifted[:,dy-1:dy]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=1)
        shifted[:,:dy] = shifted[:,dy:dy+1]
    return shifted


def _motion_blur(x, radius, sigma, angle):
    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])
    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dx = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dy = -math.ceil(((i*point[1]) / hypot) - 0.5)
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred


def motion_blur(x, severity=1):
    shape = np.array(x).shape
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = np.array(x)
    angle = np.random.uniform(-45, 45)
    x = _motion_blur(x, radius=c[0], sigma=c[1], angle=angle)
    # ----------
    if len(x.shape) < 3 or x.shape[2] < 3:
        gray = np.clip(np.array(x).transpose((0, 1)), 0, 255)
        if len(shape) >= 3 or shape[2] >=3:
            return np.stack([gray, gray, gray], axis=2)
        else:
            return gray
    else:
        return np.clip(x, 0, 255)


# --------------------------------------------------------------------------------------------------
# add snow
# --------------------------------------------------------------------------------------------------

def add_snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
    # ----------
    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome
    # snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0
    snow_layer = np.clip(snow_layer.squeeze(), 0, 1)
    snow_layer = _motion_blur(snow_layer, radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    # ----------
    # The snow layer is rounded and cropped to the img dims
    snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    snow_layer = snow_layer[:x.shape[0], :x.shape[1], :]
    # ----------
    if len(x.shape) < 3 or x.shape[2] < 3:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, x.reshape(x.shape[0], x.shape[1]) * 1.5 + 0.5)
        snow_layer = snow_layer.squeeze(-1)
    else:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(
            x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
    try:
        return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    except ValueError:
        print('ValueError for Snow, Exception handling')
        x[:snow_layer.shape[0], :snow_layer.shape[1]] += snow_layer + np.rot90(
            snow_layer, k=2)
        return np.clip(x, 0, 1) * 255


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir = os.path.join(base_path, '00_sample_images')

image_dir = os.path.join(base_path, '00_sample_images')

# image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
# print(f'num of images: {len(image_path_list)}')


# ----------
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/blue/*')))
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/yellow/*')))
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/red/*')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.png')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, '*.png')))


# --------------------------------------------------------------------------------------------------
# add snow
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
    # add snow
    # --------------------------------------------------------------------------------------------------
    for i in range(5):
        img_snow_rgb = add_snow(img, severity=i+1).astype('uint8')
        if i == 0:
            img_to_show = np.hstack([img_rgb, img_snow_rgb])
        else:
            img_to_show = np.hstack([img_to_show, img_snow_rgb])
    PIL.Image.fromarray(img_to_show.astype('uint8')).show()


