
import os
import glob
import shutil

import numpy as np
import numpy.matlib

import math
import random

import cv2
import PIL.Image

import matplotlib.pyplot as plt

import pandas as pd


####################################################################################################
# --------------------------------------------------------------------------------------------------
# correct exposure
#  - CLAHE to y in yub
# --------------------------------------------------------------------------------------------------

def correct_exposure(image):
    image = np.copy(image)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    ones = np.ones(img_yuv[...,0].shape)
    ones[img_yuv[...,0]>150]= 0.85
    img_yuv[...,0] = img_yuv[:,:,0]*ones
    # ----------
    img_yuv[...,0] = clahe.apply(img_yuv[...,0])
    img_yuv[...,0] = cv2.equalizeHist(img_yuv[...,0])
    img_yuv[...,0] = clahe.apply(img_yuv[...,0])
    # ----------
    image_res = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    image_res= cv2.fastNlMeansDenoisingColored(image_res,None,3,3,7,21)
    image_res = cv2.cvtColor(image_res, cv2.COLOR_BGR2RGB)
    return image_res


# --------------------------------------------------------------------------------------------------
# darken and brighten
# --------------------------------------------------------------------------------------------------

def change_light(image, coeff):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    image_HLS[...,1] = image_HLS[...,1]*coeff ## scale pixel values up or down for channel 1(Lightness)
    if(coeff>1):
        image_HLS[...,1][image_HLS[...,1]>255]  = 255 ##Sets all values above 255 to 255
    else:
        image_HLS[...,1][image_HLS[...,1]<0]=0
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB 


def darken(image, darkness_coeff=-1):
    if(darkness_coeff==-1):
            darkness_coeff_t = 1 - random.uniform(0,1)
    else:
        darkness_coeff_t = 1 - darkness_coeff
    image_RGB= change_light(image,darkness_coeff_t)
    return image_RGB


def brighten(image, brightness_coeff=-1):
    if(brightness_coeff==-1):
        brightness_coeff_t = 1 + random.uniform(0,1) ## coeff between 1.0 and 1.5
    else:
        brightness_coeff_t = 1 + brightness_coeff ## coeff between 1.0 and 2.0
    image_RGB= change_light(image,brightness_coeff_t)
    return image_RGB


####################################################################################################
# --------------------------------------------------------------------------------------------------
# adjust brightness, contrast
# --------------------------------------------------------------------------------------------------

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

def _brightness_contrast_adjust_non_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = img.dtype
    img = img.astype("float32")
    # ----------
    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = np.dtype("uint8")
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # ----------
    lut = np.arange(0, max_value + 1).astype("float32")
    # ----------
    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += (alpha * beta) * np.mean(img)
    # ----------
    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img


def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):
    if img.dtype == np.uint8:
        return _brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)
    return _brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# gamma transform
# --------------------------------------------------------------------------------------------------

def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    else:
        img = np.power(img, gamma)
    return img


# we perceive double the amount of light as only a fraction brighter
# (a non-linear relationship)! Furthermore, our eyes are also much more sensitive to
# changes in dark tones than brighter tones (another non-linear relationship)

# gamma correction, a translation between the sensitivity of our eyes and sensors of a camera.

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# move tone curve
# --------------------------------------------------------------------------------------------------

def move_tone_curve(img, low_y, high_y):
    """Rescales the relationship between bright and dark areas of the image by manipulating its tone curve.
    Args:
        img (numpy.ndarray): RGB or grayscale image.
        low_y (float): y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y (float): y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]
    """
    input_dtype = img.dtype
    # ----------
    if low_y < 0 or low_y > 1:
        raise ValueError("low_shift must be in range [0, 1]")
    if high_y < 0 or high_y > 1:
        raise ValueError("high_shift must be in range [0, 1]")
    # ----------
    if input_dtype != np.uint8:
        raise ValueError("Unsupported image type {}".format(input_dtype))
    # ----------
    t = np.linspace(0.0, 1.0, 256)
    # ----------
    # Defines responze of a four-point bezier curve
    def evaluate_bez(t):
        return 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t**2 * high_y + t**3
    # ----------
    evaluate_bez = np.vectorize(evaluate_bez)
    remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)
    # ----------
    return cv2.LUT(img, lut=remapping)


def move_tone_curve_rgb(img, low_r, high_r, low_g, high_g, low_b, high_b):
    r, g, b = cv2.split(img)
    t = np.linspace(0.0, 1.0, 256)
    def evaluate_bez_r(t):
        return 3 * (1 - t) ** 2 * t * low_r + 3 * (1 - t) * t**2 * high_r + t**3
    def evaluate_bez_g(t):
        return 3 * (1 - t) ** 2 * t * low_g + 3 * (1 - t) * t**2 * high_g + t**3
    def evaluate_bez_b(t):
        return 3 * (1 - t) ** 2 * t * low_b + 3 * (1 - t) * t**2 * high_b + t**3
    # ----------
    evaluate_bez_r = np.vectorize(evaluate_bez_r)
    evaluate_bez_g = np.vectorize(evaluate_bez_g)
    evaluate_bez_b = np.vectorize(evaluate_bez_b)
    remapping_r = np.rint(evaluate_bez_r(t) * 255).astype(np.uint8)
    remapping_g = np.rint(evaluate_bez_g(t) * 255).astype(np.uint8)
    remapping_b = np.rint(evaluate_bez_b(t) * 255).astype(np.uint8)
    # ----------
    r_remap = cv2.LUT(r, lut=remapping_r)
    g_remap = cv2.LUT(g, lut=remapping_g)
    b_remap = cv2.LUT(b, lut=remapping_b)
    return cv2.merge([r_remap, g_remap, b_remap])


####################################################################################################
# --------------------------------------------------------------------------------------------------
# helpers function
# --------------------------------------------------------------------------------------------------

def get_bbox_by_inv_maxarea(img, threshold=(250, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, threshold[0], threshold[1], cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    xmin, ymin, width, height = cv2.boundingRect(max_contour)
    return xmin, ymin, width, height, max_contour, contours, gray, bin_img


def get_bbox_by_maxarea(img, threshold=(128, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, threshold[0], threshold[1], cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    xmin, ymin, width, height = cv2.boundingRect(max_contour)
    return xmin, ymin, width, height, max_contour, contours, gray, bin_img


####################################################################################################
# --------------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'


# ----------
color_chart_dir = os.path.join(base_path, '00_sample_images/color_correction')
color_chart_path = os.path.join(color_chart_dir, 'reference.jpg')
# color_chart_path = os.path.join(color_chart_dir, '01.jpg')


# ----------
save_folder = os.path.join(base_path, '04_output_color_chart')

save_folder_proc = os.path.join(save_folder, 'color_chart_proc')

if os.path.exists(save_folder_proc):
    shutil.rmtree(save_folder_proc)

if not os.path.exists(save_folder_proc):
    os.makedirs(save_folder_proc, exist_ok=True)


# ----------
rgb_val_colname = ['type', 'vari', 'column', 'row', 'r', 'g', 'b']


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load colour chart
# --------------------------------------------------------------------------------------------------

color_chart_img = cv2.imread(color_chart_path)

print(color_chart_img.shape)


# --------------------------------------------------------------------------------------------------
# get bounding box coordinate and cropping
# --------------------------------------------------------------------------------------------------

# xmin, ymin, width, height, _, _, gray, bin_img = \
#     get_bbox_by_inv_maxarea(color_chart_img, threshold=(128, 255))
#
# print(f'xmin: {xmin}  ymin: {ymin}  width: {width}  height: {height}')
#
# PIL.Image.fromarray(cv2.cvtColor(bin_img.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
xmin, ymin, width, height = 560, 615, 1620, 1805

img_vis = cv2.rectangle(color_chart_img,
                        (int(xmin), int(ymin)), (int(xmin+width), int(ymin+height)), (255, 0, 0), 3)

PIL.Image.fromarray(cv2.cvtColor(img_vis.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
color_chart_img_cropped = color_chart_img[ymin:ymin+height, xmin:xmin+width]

PIL.Image.fromarray(cv2.cvtColor(color_chart_img_cropped.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# get RGB value of each color
# --------------------------------------------------------------------------------------------------

xmin, ymin = 0, 0
width = color_chart_img_cropped.shape[1]
height = color_chart_img_cropped.shape[0]

n_col, n_row = 9, 8

shift_x = [70] + [200] * (n_row - 1)
shift_y = [70] + [200] * (n_col - 1)
rec_width = 60
rec_height = rec_width

img_vis = color_chart_img_cropped.copy()
color_rec = (255, 0, 0)

rgb_val_ref = []

for co in range(n_col):
    for ro in range(n_row):
        xmin2 = xmin + sum(shift_x[:ro+1])
        ymin2 = ymin + sum(shift_y[:co+1])
        img_vis = cv2.rectangle(img_vis, (int(xmin2), int(ymin2)), (int(xmin2+rec_width), int(ymin2+rec_height)), color_rec, 3)
        b, g, r = cv2.split(color_chart_img_cropped[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width])
        rgb_val_ref.append(('ref', '', co, ro, np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2)))

PIL.Image.fromarray(cv2.cvtColor(img_vis.astype('uint8'), cv2.COLOR_BGR2RGB)).show()

print(rgb_val_ref)


# ----------
save_fpath_rgb_val_ref = os.path.join(save_folder_proc, 'rgb_val_ref.txt')
rgb_val_ref_pd = pd.DataFrame(rgb_val_ref)
rgb_val_ref_pd.columns = rgb_val_colname

rgb_val_ref_pd.to_csv(save_fpath_rgb_val_ref, sep=',', index=False)


# ----------
img_rgb = cv2.cvtColor(color_chart_img_cropped, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------------------
# generate color chart with various image processing
#  - gamma correction
# --------------------------------------------------------------------------------------------------

# ##############################
# # CHECK STEP BY STEP
# t = np.linspace(0.0, 1.0, 256)
# save_fpath_tmp = os.path.join(save_folder_proc, 'tmp')
# par_g_list = np.linspace(0.1, 3.0, 30)
#
# for par_g in par_g_list:
#     fpath = os.path.join(save_fpath_tmp, f'gamma_{par_g}.png')
#     invGamma = 1.0 / par_g
#     val = t ** invGamma
#     plt.plot(t, val)
#     plt.savefig(fpath)
#     plt.close()
# ##############################

proc_type = 'gamma_c'
rgb_val = []

save_folder_proc2 = os.path.join(save_folder_proc, proc_type)
if os.path.exists(save_folder_proc2):
    shutil.rmtree(save_folder_proc2)

if not os.path.exists(save_folder_proc2):
    os.makedirs(save_folder_proc2, exist_ok=True)

par_list = np.linspace(0.25, 2.5, 10)

for par in par_list:
    img_proc = adjust_gamma(img_rgb, par)
    suffix = format(par, '.2f')
    fpath = os.path.join(save_folder_proc2, f'color_chart_{proc_type}_{suffix}.png')
    cv2.imwrite(fpath, img_proc)
    for co in range(n_col):
        for ro in range(n_row):
            xmin2 = xmin + sum(shift_x[:ro+1])
            ymin2 = ymin + sum(shift_y[:co+1])
            b, g, r = cv2.split(img_proc[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width])
            rgb_val.append((proc_type, suffix, co, ro, np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2)))

save_fpath_rgb_val = os.path.join(save_folder_proc, f'rgb_val_{proc_type}.txt')
rgb_val_pd = pd.DataFrame(rgb_val)
rgb_val_pd.columns = rgb_val_colname
rgb_val_pd.to_csv(save_fpath_rgb_val, sep=',', index=False)


# --------------------------------------------------------------------------------------------------
# generate color chart with various image processing
#  - move tone curve
# --------------------------------------------------------------------------------------------------

##############################
# CHECK STEP BY STEP
# r, g, b = cv2.split(img_rgb)

t = np.linspace(0.0, 1.0, 256)
def evaluate_bez_r(t):
    return 3 * (1 - t) ** 2 * t * par_l_r + 3 * (1 - t) * t ** 2 * par_h_r + t ** 3
def evaluate_bez_g(t):
    return 3 * (1 - t) ** 2 * t * par_l_g + 3 * (1 - t) * t ** 2 * par_h_g + t ** 3
def evaluate_bez_b(t):
    return 3 * (1 - t) ** 2 * t * par_l_b + 3 * (1 - t) * t ** 2 * par_h_b + t ** 3

par_l = 0.4
par_h = 0.6
bez = 3 * (1 - t) ** 2 * t * par_l + 3 * (1 - t) * t ** 2 * par_h + t ** 3

plt.plot(t, bez)
plt.show()

t = np.linspace(0.0, 1.0, 256)
save_fpath_tmp = os.path.join(save_folder, 'tmp')
par_l_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
par_h_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for par_l in par_l_list:
    for par_h in par_h_list:
        if par_l < par_h:
            print(f'{par_l} - {par_h}')
            fpath = os.path.join(save_fpath_tmp, f'curve_{par_l}_{par_h}.png')
            bez = 3 * (1 - t) ** 2 * t * par_l + 3 * (1 - t) * t ** 2 * par_h + t ** 3
            # bez = 4 * (1 - t) ** 2 * t * par_l + 4 * (1 - t) * t ** 2 * par_h + t ** 4
            # bez = 2 * (1 - t) ** 2 * t * par_l + 2 * (1 - t) * t ** 2 * par_h + t ** 2
            # bez = 5 * (1 - t) ** 2 * t * par_l + 5 * (1 - t) * t ** 2 * par_h + t ** 5
            plt.plot(t, bez)
            plt.savefig(fpath)
            plt.close()

#
# par_l_r, par_h_r = 0.4, 0.6
# par_l_g, par_h_g = 0.1, 0.9
# par_l_b, par_h_b = 0.2, 0.8
#
# evaluate_bez_r = np.vectorize(evaluate_bez_r)
# evaluate_bez_g = np.vectorize(evaluate_bez_g)
# evaluate_bez_b = np.vectorize(evaluate_bez_b)
#
# remapping_r = np.rint(evaluate_bez_r(t) * 255).astype(np.uint8)
# remapping_g = np.rint(evaluate_bez_g(t) * 255).astype(np.uint8)
# remapping_b = np.rint(evaluate_bez_b(t) * 255).astype(np.uint8)
#
# r_remap = cv2.LUT(r, lut=remapping_r)
# g_remap = cv2.LUT(g, lut=remapping_g)
# b_remap = cv2.LUT(b, lut=remapping_b)
# img_proc = cv2.merge([r_remap, g_remap, b_remap])
# PIL.Image.fromarray(img_proc.astype('uint8')).show()
# ##############################

proc_type = 'tonecurve'
rgb_val = []

save_folder_proc2 = os.path.join(save_folder_proc, proc_type)
if os.path.exists(save_folder_proc2):
    shutil.rmtree(save_folder_proc2)

if not os.path.exists(save_folder_proc2):
    os.makedirs(save_folder_proc2, exist_ok=True)

base_l_h = [(0.2, 0.6), (0.2, 0.8), (0.4, 0.6), (0.4, 0.8), (0.6, 0.8)]
par_r_list, par_g_list, par_b_list = base_l_h, base_l_h, base_l_h

for par_r in par_r_list:
    for par_g in par_g_list:
        for par_b in par_b_list:
            img_proc = move_tone_curve_rgb(img_rgb, par_r[0], par_r[1], par_g[0], par_g[1], par_b[0], par_b[1])
            suffix = f'{str(par_r[0])}_{str(par_r[1])}_{str(par_g[0])}_{str(par_g[1])}_{str(par_b[0])}_{str(par_b[1])}'
            fpath = os.path.join(save_folder_proc2, f'color_chart_{proc_type}_{suffix}.png')
            cv2.imwrite(fpath, img_proc)
            for co in range(n_col):
                for ro in range(n_row):
                    xmin2 = xmin + sum(shift_x[:ro+1])
                    ymin2 = ymin + sum(shift_y[:co+1])
                    b, g, r = cv2.split(img_proc[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width])
                    rgb_val.append((proc_type, suffix, co, ro, np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2)))

save_fpath_rgb_val = os.path.join(save_folder_proc, f'rgb_val_{proc_type}.txt')
rgb_val_pd = pd.DataFrame(rgb_val)
rgb_val_pd.columns = rgb_val_colname
rgb_val_pd.to_csv(save_fpath_rgb_val, sep=',', index=False)


# --------------------------------------------------------------------------------------------------
# generate color chart with various image processing
#  - gamma correction + move tone curve
# --------------------------------------------------------------------------------------------------

proc_type = 'gamma + tonecurve'
rgb_val = []

save_folder_proc2 = os.path.join(save_folder_proc, proc_type)
if os.path.exists(save_folder_proc2):
    shutil.rmtree(save_folder_proc2)

if not os.path.exists(save_folder_proc2):
    os.makedirs(save_folder_proc2, exist_ok=True)

base_l_h = [(0.2, 0.6), (0.2, 0.8), (0.4, 0.6), (0.4, 0.8), (0.6, 0.8)]
par_r_list, par_g_list, par_b_list = base_l_h, base_l_h, base_l_h

for par_r in par_r_list:
    for par_g in par_g_list:
        for par_b in par_b_list:
            img_proc = move_tone_curve_rgb(img_rgb, par_r[0], par_r[1], par_g[0], par_g[1], par_b[0], par_b[1])
            suffix = f'{str(par_r[0])}_{str(par_r[1])}_{str(par_g[0])}_{str(par_g[1])}_{str(par_b[0])}_{str(par_b[1])}'
            fpath = os.path.join(save_folder_proc2, f'color_chart_{proc_type}_{suffix}.png')
            cv2.imwrite(fpath, img_proc)
            for co in range(n_col):
                for ro in range(n_row):
                    xmin2 = xmin + sum(shift_x[:ro+1])
                    ymin2 = ymin + sum(shift_y[:co+1])
                    b, g, r = cv2.split(img_proc[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width])
                    rgb_val.append((proc_type, suffix, co, ro, np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2)))

save_fpath_rgb_val = os.path.join(save_folder_proc, f'rgb_val_{proc_type}.txt')
rgb_val_pd = pd.DataFrame(rgb_val)
rgb_val_pd.columns = rgb_val_colname
rgb_val_pd.to_csv(save_fpath_rgb_val, sep=',', index=False)
