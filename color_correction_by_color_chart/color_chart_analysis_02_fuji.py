
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

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

pd.set_option('display.max_rows', 80)
pd.set_option('display.max_columns', 20)


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


# --------------------------------------------------------------------------------------------------
# tone curve adjustment
# --------------------------------------------------------------------------------------------------
def move_tone_curve(img, low_0, high_0, low_1, high_1, low_2, high_2):
    val0, val1, val2 = cv2.split(img)
    t = np.linspace(0.0, 1.0, 256)
    def evaluate_bez_0(t):
        return 3 * (1 - t) ** 2 * t * low_0 + 3 * (1 - t) * t ** 2 * high_0 + t ** 3
    def evaluate_bez_1(t):
        return 3 * (1 - t) ** 2 * t * low_1 + 3 * (1 - t) * t ** 2 * high_1 + t ** 3
    def evaluate_bez_2(t):
        return 3 * (1 - t) ** 2 * t * low_2 + 3 * (1 - t) * t ** 2 * high_2 + t ** 3
    # ----------
    evaluate_bez_0 = np.vectorize(evaluate_bez_0)
    evaluate_bez_1 = np.vectorize(evaluate_bez_1)
    evaluate_bez_2 = np.vectorize(evaluate_bez_2)
    remapping_0 = np.rint(evaluate_bez_1(t) * 255).astype(np.uint8)
    remapping_1 = np.rint(evaluate_bez_1(t) * 255).astype(np.uint8)
    remapping_2 = np.rint(evaluate_bez_2(t) * 255).astype(np.uint8)
    # ----------
    val0_remap = cv2.LUT(val0, lut=remapping_0)
    val1_remap = cv2.LUT(val1, lut=remapping_1)
    val2_remap = cv2.LUT(val2, lut=remapping_2)
    return cv2.merge([val0_remap, val1_remap, val2_remap])


def move_tone_curve_hsv(img, low_0, high_0, low_1, high_1, low_2, high_2):
    val0, val1, val2 = cv2.split(img)
    t = np.linspace(0.0, 1.0, 256)
    t2 = np.linspace(0.0, 1.0, 180)
    def evaluate_bez_0(t2):
        return 3 * (1 - t2) ** 2 * t2 * low_0 + 3 * (1 - t2) * t2 ** 2 * high_0 + t2 ** 3
    def evaluate_bez_1(t):
        return 3 * (1 - t) ** 2 * t * low_1 + 3 * (1 - t) * t ** 2 * high_1 + t ** 3
    def evaluate_bez_2(t):
        return 3 * (1 - t) ** 2 * t * low_2 + 3 * (1 - t) * t ** 2 * high_2 + t ** 3
    # ----------
    evaluate_bez_0 = np.vectorize(evaluate_bez_0)
    evaluate_bez_1 = np.vectorize(evaluate_bez_1)
    evaluate_bez_2 = np.vectorize(evaluate_bez_2)
    remapping_0 = np.rint(evaluate_bez_0(t2) * 179).astype(np.uint8)
    remapping_1 = np.rint(evaluate_bez_1(t) * 255).astype(np.uint8)
    remapping_2 = np.rint(evaluate_bez_2(t) * 255).astype(np.uint8)
    # ----------
    val0_remap = cv2.LUT(val0, lut=remapping_0)
    val1_remap = cv2.LUT(val1, lut=remapping_1)
    val2_remap = cv2.LUT(val2, lut=remapping_2)
    return cv2.merge([val0_remap, val1_remap, val2_remap])


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


def get_palette_values(img, n_row, n_col, xmin, ymin, shift_x, shift_y, rec_width, rec_height, input_mode='bgr', is_vis=False):
    if input_mode == 'rgb':
        img_vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    elif input_mode == 'hsv':
        img_vis = cv2.cvtColor(img, cv2.COLOR_HSV2BGR).copy()
    elif input_mode == 'lab':
        img_vis = cv2.cvtColor(img, cv2.COLOR_Lab2BGR).copy()
    color_rect = (255, 0, 0)
    val_list = []
    for ro in range(n_row):
        for co in range(n_col):
            xmin2 = xmin + sum(shift_x[:co+1])
            ymin2 = ymin + sum(shift_y[:ro+1])
            if is_vis:
                img_vis = cv2.rectangle(img_vis, (int(xmin2), int(ymin2)), (int(xmin2+rec_width), int(ymin2+rec_height)), color_rect, 3)
            img_tmp = img[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width]
            if input_mode == 'bgr':
                b, g, r = cv2.split(img_tmp)
                labl, laba, labb = cv2.split(cv2.cvtColor(img_tmp, cv2.COLOR_BGR2Lab))
                hsvh, hsvs, hsvv = cv2.split(cv2.cvtColor(img_tmp, cv2.COLOR_BGR2HSV))
            elif input_mode == 'rgb':
                img_tmp2 = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)
                r, g, b = cv2.split(img_tmp)
                labl, laba, labb = cv2.split(cv2.cvtColor(img_tmp2, cv2.COLOR_BGR2Lab))
                hsvh, hsvs, hsvv = cv2.split(cv2.cvtColor(img_tmp2, cv2.COLOR_BGR2HSV))
            elif input_mode == 'lab':
                img_tmp2 = cv2.cvtColor(img_tmp, cv2.COLOR_Lab2BGR)
                b, g, r = cv2.split(img_tmp2)
                labl, laba, labb = cv2.split(img_tmp)
                hsvh, hsvs, hsvv = cv2.split(cv2.cvtColor(img_tmp2, cv2.COLOR_BGR2HSV))
            elif input_mode == 'hsv':
                img_tmp2 = cv2.cvtColor(img_tmp, cv2.COLOR_HSV2BGR)
                b, g, r = cv2.split(img_tmp2)
                labl, laba, labb = cv2.split(cv2.cvtColor(img_tmp2, cv2.COLOR_BGR2Lab))
                hsvh, hsvs, hsvv = cv2.split(img_tmp)
            val_list.append(
                (
                    ro, co,
                    np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                    np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
                    np.round(hsvh.mean(), 2), np.round(hsvs.mean(), 2), np.round(hsvv.mean(), 2),
                )
            )
    val_pd = pd.DataFrame(val_list)
    val_pd.columns = ['row', 'column', 'r', 'g', 'b', 'labl', 'laba', 'labb', 'hsvh', 'hsvs', 'hsvv']
    return val_pd

def compare_values(ref, proc, test):
    obj_col0 = ['row', 'column']
    obj_col_rgb = ['r', 'g', 'b']
    obj_col_lab = ['labl', 'laba', 'labb']
    obj_col_hsv = ['hsvh', 'hsvs', 'hsvv']
    ref_rgb = ref[obj_col_rgb]
    proc_rgb = proc[obj_col_rgb]
    test_rgb = test[obj_col_rgb]
    ref_lab = ref[obj_col_lab]
    proc_lab = proc[obj_col_lab]
    test_lab = test[obj_col_lab]
    ref_hsv = ref[obj_col_hsv]
    proc_hsv = proc[obj_col_hsv]
    test_hsv = test[obj_col_hsv]
    ref_rgb.columns = ['r_ref', 'g_ref', 'b_ref']
    proc_rgb.columns = ['r_proc', 'g_proc', 'b_proc']
    test_rgb.columns = ['r_test', 'g_test', 'b_test']
    ref_lab.columns = ['labl_ref', 'laba_ref', 'labb_ref']
    proc_lab.columns = ['labl_proc', 'laba_proc', 'labb_proc']
    test_lab.columns = ['labl_test', 'laba_test', 'labb_test']
    ref_hsv.columns = ['hsvh_ref', 'hsvs_ref', 'hsvv_ref']
    proc_hsv.columns = ['hsvh_proc', 'hsvs_proc', 'hsvv_proc']
    test_hsv.columns = ['hsvh_test', 'hsvs_test', 'hsvv_test']
    dif_ref_rgb = pd.DataFrame(np.array([
        np.round(ref.r.values - test.r.values, 2),
        np.round(ref.g.values - test.b.values, 2),
        np.round(ref.g.values - test.b.values, 2)]).T)
    dif_ref_lab = pd.DataFrame(np.array([
        np.round(ref.labl.values - test.labl.values, 2),
        np.round(ref.laba.values - test.laba.values, 2),
        np.round(ref.labb.values - test.labb.values, 2)]).T)
    dif_ref_hsv = pd.DataFrame(np.array([
        np.round(ref.hsvh.values - test.hsvh.values, 2),
        np.round(ref.hsvs.values - test.hsvs.values, 2),
        np.round(ref.hsvv.values - test.hsvv.values, 2)]).T)
    dif_proc_rgb = pd.DataFrame(np.array([
        np.round(proc.r.values - test.r.values, 2),
        np.round(proc.g.values - test.b.values, 2),
        np.round(proc.g.values - test.b.values, 2)]).T)
    dif_proc_lab = pd.DataFrame(np.array([
        np.round(proc.labl.values - test.labl.values, 2),
        np.round(proc.laba.values - test.laba.values, 2),
        np.round(proc.labb.values - test.labb.values, 2)]).T)
    dif_proc_hsv = pd.DataFrame(np.array([
        np.round(proc.hsvh.values - test.hsvh.values, 2),
        np.round(proc.hsvs.values - test.hsvs.values, 2),
        np.round(proc.hsvv.values - test.hsvv.values, 2)]).T)
    dif_ref_rgb.columns = ['dif_r_ref', 'dif_g_ref', 'dif_b_ref']
    dif_ref_lab.columns = ['dif_labl_ref', 'dif_laba_ref', 'dif_labb_ref']
    dif_ref_hsv.columns = ['dif_hsvh_ref', 'dif_hsvs_ref', 'dif_hsvv_ref']
    dif_proc_rgb.columns = ['dif_r_proc', 'dif_g_proc', 'dif_b_proc']
    dif_proc_lab.columns = ['dif_labl_proc', 'dif_laba_proc', 'dif_labb_proc']
    dif_proc_hsv.columns = ['dif_hsvh_proc', 'dif_hsvs_proc', 'dif_hsvv_proc']
    output_pd = pd.concat([
        ref[obj_col0],
        ref_rgb, proc_rgb, test_rgb, dif_ref_rgb, dif_proc_rgb,
        ref_lab, proc_lab, test_lab, dif_ref_lab, dif_proc_lab,
        ref_hsv, proc_hsv, test_hsv, dif_ref_hsv, dif_proc_hsv,
        ], axis=1)
    return output_pd


####################################################################################################
# --------------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing/color_correction_by_color_chart'


# ----------
color_chart_dir = os.path.join(base_path, '00_img_color_chart')
color_chart_path = os.path.join(color_chart_dir, 'reference.jpg')
# color_chart_path = os.path.join(color_chart_dir, '01.jpg')


# ----------
save_folder = os.path.join(base_path, '04_output_color_chart')

save_folder_ref = os.path.join(save_folder, 'color_chart_ref')
save_folder_test = os.path.join(save_folder, 'color_chart_test')
save_folder_val = os.path.join(save_folder, 'val')
save_folder_val_result = os.path.join(save_folder, 'val_result')
save_folder_result = os.path.join(save_folder, 'result_plot_img')


# if os.path.exists(save_folder_proc):
#     shutil.rmtree(save_folder_proc)
#
# if not os.path.exists(save_folder_proc):
#     os.makedirs(save_folder_proc, exist_ok=True)

# ----------
save_fpath_val_ref = os.path.join(save_folder_val, 'val_ref.txt')
save_fpath_val_proc = os.path.join(save_folder_val, f'val_gamma_tonecurve_proc.txt')
save_fpath_val_proc2 = os.path.join(save_folder_val, f'val_gamma_tonecurve_proc2.txt')
save_fpath_val_proc3 = os.path.join(save_folder_val, f'val_gamma_tonecurve_proc3.txt')
save_fpath_val_proc4 = os.path.join(save_folder_val, f'val_gamma_tonecurve_proc4.txt')


# ----------
val_colname = [
    'type', 'vari', 'gamma',
    'tone_0_low', 'tone_0_high',
    'tone_1_low', 'tone_1_high',
    'tone_2_low', 'tone_2_high',
    'row', 'column',
    'r', 'g', 'b',
    'labl', 'laba', 'labb',
    'hsvh', 'hsvs', 'hsvv'
]


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load colour chart
# --------------------------------------------------------------------------------------------------

cc_img_bgr = cv2.imread(color_chart_path)

print(cc_img_bgr.shape)


# --------------------------------------------------------------------------------------------------
# get bounding box coordinate and cropping
# --------------------------------------------------------------------------------------------------

# xmin, ymin, width, height, _, _, gray, bin_img = \
#     get_bbox_by_inv_maxarea(cc_img_bgr, threshold=(128, 255))
#
# print(f'xmin: {xmin}  ymin: {ymin}  width: {width}  height: {height}')
#
# PIL.Image.fromarray(cv2.cvtColor(bin_img.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
# xmin, ymin, width, height = 560, 615, 1620, 1805
xmin, ymin, width, height = 560-180, 615-180, 1620+300, 1805+500

img_vis = cv2.rectangle(cc_img_bgr.copy(),
                        (int(xmin), int(ymin)), (int(xmin+width), int(ymin+height)), (255, 0, 0), 3)

PIL.Image.fromarray(cv2.cvtColor(img_vis.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
# PIL.Image.fromarray(cv2.cvtColor(cc_img_bgr.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
cc_img_bgr_cr = cc_img_bgr[ymin:ymin+height, xmin:xmin+width]

PIL.Image.fromarray(cv2.cvtColor(cc_img_bgr_cr.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
# save original color chart and cropped
cv2.imwrite(os.path.join(save_folder_ref, 'color_chart_original.png'), cc_img_bgr)
cv2.imwrite(os.path.join(save_folder_ref, 'color_chart_cropped.png'), cc_img_bgr_cr)


# --------------------------------------------------------------------------------------------------
# get RGB/Lab/HSV value of each color
# --------------------------------------------------------------------------------------------------

xmin, ymin = 0, 0
width_ref = cc_img_bgr_cr.shape[1]
height_ref = cc_img_bgr_cr.shape[0]

n_row, n_col = 9, 8

shift_x = [70+200] + [200] * (n_col - 1)
shift_y = [70+180] + [200] * (n_row - 1)
rec_width = 60
rec_height = rec_width

img_vis = cc_img_bgr_cr.copy()
color_rect = (255, 0, 0)

val_ref = []

for ro in range(n_row):
    for co in range(n_col):
        xmin2 = xmin + sum(shift_x[:co+1])
        ymin2 = ymin + sum(shift_y[:ro+1])
        img_vis = cv2.rectangle(img_vis, (int(xmin2), int(ymin2)), (int(xmin2+rec_width), int(ymin2+rec_height)), color_rect, 3)
        img_tmp = cc_img_bgr_cr[ymin2:ymin2 + rec_height, xmin2:xmin2 + rec_width]
        b, g, r = cv2.split(img_tmp)
        labl, laba, labb = cv2.split(cv2.cvtColor(img_tmp, cv2.COLOR_BGR2LAB))
        hsvh, hsvs, hsvv = cv2.split(cv2.cvtColor(img_tmp, cv2.COLOR_BGR2HSV))
        val_ref.append(('ref', 'ref',
                        np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan,
                        ro, co,
                        np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                        np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
                        np.round(hsvh.mean(), 2), np.round(hsvs.mean(), 2), np.round(hsvv.mean(), 2))
                       )

PIL.Image.fromarray(cv2.cvtColor(img_vis.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
print(val_ref)


# ----------
val_ref_pd = pd.DataFrame(val_ref)
val_ref_pd.columns = val_colname
val_ref_pd.to_csv(save_fpath_val_ref, sep=',', index=False)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# generate color chart with various image processing
#  1.  gamma correction + tone curve for RGB
# --------------------------------------------------------------------------------------------------

par_gamma_list = np.linspace(0.75, 2.0, 6)

base_l = np.linspace(0.2, 0.8, 4)
base_h = base_l
par_r_list = []
for l in base_l:
    for h in base_h:
        if l < h:
            par_r_list.append((l, h))

par_g_list, par_b_list = par_r_list, par_r_list

print(f'{len(par_gamma_list)} - {len(par_r_list)}')


# ----------
val_proc = []

for par_gamma in par_gamma_list:
    img_proc_gamma = adjust_gamma(cc_img_bgr_cr, par_gamma)
    for par_r in par_r_list:
        for par_g in par_g_list:
            for par_b in par_b_list:
                # if par_r == par_g and par_r == par_b:
                print(f"gamma: {format(par_gamma, '.2f')}  r: ({format(par_r[0], '.2f')}, {format(par_r[1], '.2f')})  g: ({format(par_g[0], '.2f')}, {format(par_g[1], '.2f')})  b: ({format(par_b[0], '.2f')}, {format(par_b[1], '.2f')})")
                img_proc = move_tone_curve(img_proc_gamma, par_b[0], par_b[1], par_g[0], par_g[1], par_r[0], par_r[1])
                vari = f"{format(par_gamma, '.2f')}_{format(par_r[0], '.2f')}_{format(par_r[1], '.2f')}_{format(par_g[0], '.2f')}_{format(par_g[1], '.2f')}_{format(par_b[0], '.2f')}_{format(par_b[1], '.2f')}"
                # fpath = os.path.join(save_folder_proc, f'color_chart_gamma_tonecurve_proc_{vari}.png')
                # cv2.imwrite(fpath, img_proc)
                for ro in range(n_row):
                    for co in range(n_col):
                        xmin2 = xmin + sum(shift_x[:co + 1])
                        ymin2 = ymin + sum(shift_y[:ro + 1])
                        img_proc_cr = img_proc[ymin2:ymin2 + rec_height, xmin2:xmin2 + rec_width]
                        b, g, r = cv2.split(img_proc_cr)
                        labl, laba, labb = cv2.split(cv2.cvtColor(img_proc_cr, cv2.COLOR_BGR2Lab))
                        hsvh, hsvs, hsvv = cv2.split(cv2.cvtColor(img_proc_cr, cv2.COLOR_BGR2HSV))
                        val_proc.append(('gamma_tonecurve_proc', vari,
                                         np.round(par_gamma, 2),
                                         np.round(par_r[0], 2), np.round(par_r[1], 2),
                                         np.round(par_g[0], 2), np.round(par_g[1], 2),
                                         np.round(par_b[0], 2), np.round(par_b[1], 2),
                                         ro, co,
                                         np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                                         np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
                                         np.round(hsvh.mean(), 2), np.round(hsvs.mean(), 2), np.round(hsvv.mean(), 2))
                                        )

val_proc_pd = pd.DataFrame(val_proc)
val_proc_pd.columns = val_colname
val_proc_pd.to_csv(save_fpath_val_proc, sep=',', index=False)


# --------------------------------------------------------------------------------------------------
# generate color chart with various image processing
#  2.  gamma correction + tone curve for LAB
# --------------------------------------------------------------------------------------------------

par_gamma_list = np.linspace(0.75, 2.0, 6)

base_l = np.linspace(0.2, 0.8, 4)
base_h = base_l
par_labl_list = []
for l in base_l:
    for h in base_h:
        if l < h:
            par_labl_list.append((l, h))

par_laba_list, par_labb_list = par_labl_list, par_labl_list

print(f'{len(par_gamma_list)} - {len(par_labl_list)}')


# ----------
val_proc2 = []

for par_gamma in par_gamma_list:
    img_proc_gamma = adjust_gamma(cc_img_bgr_cr, par_gamma)
    for par_labl in par_labl_list:
        for par_laba in par_laba_list:
            for par_labb in par_labb_list:
                # if par_labl == par_laba and par_laba == par_labb:
                print(f"gamma: {format(par_gamma, '.2f')}  l: ({format(par_labl[0], '.2f')}, {format(par_labl[1], '.2f')})  a: ({format(par_laba[0], '.2f')}, {format(par_laba[1], '.2f')})  labb: ({format(par_labb[0], '.2f')}, {format(par_labb[1], '.2f')})")
                img_proc = move_tone_curve(cv2.cvtColor(img_proc_gamma, cv2.COLOR_BGR2Lab), par_labl[0], par_labl[1], par_laba[0], par_laba[1], par_labb[0], par_labb[1])
                vari = f"{format(par_gamma, '.2f')}_{format(par_labl[0], '.2f')}_{format(par_labl[1], '.2f')}_{format(par_laba[0], '.2f')}_{format(par_laba[1], '.2f')}_{format(par_labb[0], '.2f')}_{format(par_labb[1], '.2f')}"
                # fpath = os.path.join(save_folder_proc2, f'color_chart_gamma_tonecurve_proc2_{vari}.png')
                # cv2.imwrite(fpath, cv2.cvtColor(img_proc, cv2.COLOR_Lab2BGR))
                for ro in range(n_row):
                    for co in range(n_col):
                        xmin2 = xmin + sum(shift_x[:ro + 1])
                        ymin2 = ymin + sum(shift_y[:co + 1])
                        img_proc_cr = img_proc[ymin2:ymin2 + rec_height, xmin2:xmin2 + rec_width]
                        img_proc_cr_bgr = cv2.cvtColor(img_proc_cr, cv2.COLOR_Lab2BGR)
                        b, g, r = cv2.split(img_proc_cr_bgr)
                        labl, laba, labb = cv2.split(img_proc_cr)
                        hsvh, hsvs, hsvv = cv2.split(cv2.cvtColor(img_proc_cr_bgr, cv2.COLOR_BGR2HSV))
                        val_proc2.append(('gamma_tonecurve_proc2', vari,
                                          np.round(par_gamma, 2),
                                          np.round(par_labl[0], 2), np.round(par_labl[1], 2),
                                          np.round(par_laba[0], 2), np.round(par_laba[1], 2),
                                          np.round(par_labb[0], 2), np.round(par_labb[1], 2),
                                          ro, co,
                                          np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                                          np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
                                          np.round(hsvh.mean(), 2), np.round(hsvs.mean(), 2), np.round(hsvv.mean(), 2))
                                         )

val_proc2_pd = pd.DataFrame(val_proc2)
val_proc2_pd.columns = val_colname
val_proc2_pd.to_csv(save_fpath_val_proc2, sep=',', index=False)


# --------------------------------------------------------------------------------------------------
# generate color chart with various image processing
#  3.  gamma correction + tone curve for HSV
# --------------------------------------------------------------------------------------------------

par_gamma_list = np.linspace(0.75, 2.0, 6)

base_l = np.linspace(0.2, 0.8, 4)
base_h = base_l
par_hsvh_list = []
for l in base_l:
    for h in base_h:
        if l < h:
            par_hsvh_list.append((l, h))

par_hsvs_list, par_hsvv_list = par_hsvh_list, par_hsvh_list

print(f'{len(par_gamma_list)} - {len(par_labl_list)}')


# ----------
val_proc3 = []

for par_gamma in par_gamma_list:
    img_proc_gamma = adjust_gamma(cc_img_bgr_cr, par_gamma)
    for par_hsvh in par_hsvh_list:
        for par_hsvs in par_hsvs_list:
            for par_hsvv in par_hsvv_list:
                print(f"gamma: {format(par_gamma, '.2f')}  hsvh: ({format(par_hsvh[0], '.2f')}, {format(par_hsvh[1], '.2f')})  hsvs: ({format(par_hsvs[0], '.2f')}, {format(par_hsvs[1], '.2f')})  hsvv: ({format(par_hsvv[0], '.2f')}, {format(par_hsvv[1], '.2f')})")
                img_proc = move_tone_curve_hsv(cv2.cvtColor(img_proc_gamma, cv2.COLOR_BGR2HSV), par_hsvh[0], par_hsvh[1], par_hsvs[0], par_hsvs[1], par_hsvv[0], par_hsvv[1])
                vari = f"{format(par_gamma, '.2f')}_{format(par_hsvh[0], '.2f')}_{format(par_hsvh[1], '.2f')}_{format(par_hsvs[0], '.2f')}_{format(par_hsvs[1], '.2f')}_{format(par_hsvv[0], '.2f')}_{format(par_hsvv[1], '.2f')}"
                # fpath = os.path.join(save_folder_proc3, f'color_chart_gamma_tonecurve_proc3_{vari}.png')
                # cv2.imwrite(fpath, cv2.cvtColor(img_proc, cv2.COLOR_HSV2BGR))
                for ro in range(n_row):
                    for co in range(n_col):
                        xmin2 = xmin + sum(shift_x[:ro + 1])
                        ymin2 = ymin + sum(shift_y[:co + 1])
                        img_proc_cr = img_proc[ymin2:ymin2 + rec_height, xmin2:xmin2 + rec_width]
                        img_proc_cr_bgr = cv2.cvtColor(img_proc_cr, cv2.COLOR_HSV2BGR)
                        b, g, r = cv2.split(img_proc_cr_bgr)
                        labl, laba, labb = cv2.split(cv2.cvtColor(img_proc_cr_bgr, cv2.COLOR_BGR2Lab))
                        hsvh, hsvs, hsvv = cv2.split(img_proc_cr)
                        val_proc3.append(('gamma_tonecurve_proc3', vari,
                                          np.round(par_gamma, 2),
                                          np.round(par_hsvh[0], 2), np.round(par_hsvh[1], 2),
                                          np.round(par_hsvs[0], 2), np.round(par_hsvs[1], 2),
                                          np.round(par_hsvv[0], 2), np.round(par_hsvv[1], 2),
                                          ro, co,
                                          np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                                          np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
                                          np.round(hsvh.mean(), 2), np.round(hsvs.mean(), 2), np.round(hsvv.mean(), 2))
                                         )

val_proc3_pd = pd.DataFrame(val_proc3)
val_proc3_pd.columns = val_colname
val_proc3_pd.to_csv(save_fpath_val_proc3, sep=',', index=False)


# --------------------------------------------------------------------------------------------------
# generate color chart with various image processing
#  4.  only gamma + A,B
# --------------------------------------------------------------------------------------------------

par_gamma_list = [0.9, 1.0, 1.1, 1.2]
par_labl_list = [(0.4, 0.6)]
base_l = np.linspace(0.0, 1.0, 11)
base_h = base_l
par_laba_list = []
for l in base_l:
    for h in base_h:
        par_laba_list.append((l, h))
par_labb_list = par_laba_list

# par_gamma_list = [1.0]
#
# # base_l = np.linspace(0.2, 0.8, 4)
# base_l = np.linspace(0.0, 1.0, 11)
# base_h = base_l
# par_laba_list = []
# for l in base_l:
#     for h in base_h:
#         par_laba_list.append((l, h))
# par_labb_list = par_laba_list

# ----------
vari_cnt = 0
for par_gamma in par_gamma_list:
    for par_labl in par_labl_list:
        for par_laba in par_laba_list:
            for par_labb in par_labb_list:
                vari_cnt += 1

val_proc4 = np.zeros((vari_cnt * n_row * n_col, 18))
print(val_proc4.shape)

val_proc4_2 = []

cnt = -1
for par_gamma in par_gamma_list:
    img_proc_gamma = adjust_gamma(cc_img_bgr_cr, par_gamma)
    # img_proc_gamma = cc_img_bgr_cr.copy()
    for par_labl in par_labl_list:
        for par_laba in par_laba_list:
            for par_labb in par_labb_list:
                # if par_labl == par_laba and par_laba == par_labb:
                print(f"{cnt}:  gamma: {format(par_gamma, '.2f')}  l: ({format(par_labl[0], '.2f')}, {format(par_labl[1], '.2f')})  a: ({format(par_laba[0], '.2f')}, {format(par_laba[1], '.2f')})  labb: ({format(par_labb[0], '.2f')}, {format(par_labb[1], '.2f')})")
                img_proc = move_tone_curve(cv2.cvtColor(img_proc_gamma, cv2.COLOR_BGR2Lab), par_labl[0], par_labl[1], par_laba[0], par_laba[1], par_labb[0], par_labb[1])
                vari = f"{format(par_gamma, '.2f')}_{format(par_labl[0], '.2f')}_{format(par_labl[1], '.2f')}_{format(par_laba[0], '.2f')}_{format(par_laba[1], '.2f')}_{format(par_labb[0], '.2f')}_{format(par_labb[1], '.2f')}"
                # fpath = os.path.join(save_folder_proc2, f'color_chart_gamma_tonecurve_proc2_{vari}.png')
                # cv2.imwrite(fpath, cv2.cvtColor(img_proc, cv2.COLOR_Lab2BGR))
                val_proc4_2 = val_proc4_2 + [('gamma_tonecurve_proc4', vari) for i in range(n_row * n_col)]
                for ro in range(n_row):
                    for co in range(n_col):
                        xmin2 = xmin + sum(shift_x[:ro + 1])
                        ymin2 = ymin + sum(shift_y[:co + 1])
                        img_proc_cr = img_proc[ymin2:ymin2 + rec_height, xmin2:xmin2 + rec_width]
                        img_proc_cr_bgr = cv2.cvtColor(img_proc_cr, cv2.COLOR_Lab2BGR)
                        b, g, r = cv2.split(img_proc_cr_bgr)
                        labl, laba, labb = cv2.split(img_proc_cr)
                        hsvh, hsvs, hsvv = cv2.split(cv2.cvtColor(img_proc_cr_bgr, cv2.COLOR_BGR2HSV))
                        # val_proc4.append(('gamma_tonecurve_proc4', vari,
                        #                   np.round(par_gamma, 2),
                        #                   np.round(par_labl[0], 2), np.round(par_labl[1], 2),
                        #                   np.round(par_laba[0], 2), np.round(par_laba[1], 2),
                        #                   np.round(par_labb[0], 2), np.round(par_labb[1], 2),
                        #                   ro, co,
                        #                   np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                        #                   np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
                        #                   np.round(hsvh.mean(), 2), np.round(hsvs.mean(), 2), np.round(hsvv.mean(), 2))
                        #                  )
                        cnt += 1
                        val_proc4[cnt] = [
                                          np.round(par_gamma, 2),
                                          np.round(par_labl[0], 2), np.round(par_labl[1], 2),
                                          np.round(par_laba[0], 2), np.round(par_laba[1], 2),
                                          np.round(par_labb[0], 2), np.round(par_labb[1], 2),
                                          ro, co,
                                          np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                                          np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
                                          np.round(hsvh.mean(), 2), np.round(hsvs.mean(), 2), np.round(hsvv.mean(), 2)]


val_proc4_pd = pd.DataFrame(val_proc4)
val_proc4_2_pd = pd.DataFrame(val_proc4_2)

val_proc4_pd = pd.concat([val_proc4_2_pd, val_proc4_pd], axis=1, ignore_index=True)

val_proc4_pd.columns = val_colname
val_proc4_pd.to_csv(save_fpath_val_proc4, sep=',', index=False)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# test setting
# --------------------------------------------------------------------------------------------------

# img_test_path = os.path.join(color_chart_dir, '01.jpg')
# img_test_path = os.path.join(color_chart_dir, '02.jpg')
img_test_path = os.path.join(color_chart_dir, '03.jpg')

# save_fpath_val_test = os.path.join(save_folder_val, 'val_test_01.txt')
# save_fpath_val_test = os.path.join(save_folder_val, 'val_test_02.txt')
save_fpath_val_test = os.path.join(save_folder_val, 'val_test_03.txt')

# save_fpath_img_orig = os.path.join(save_folder_test, 'test01_orig.png')
# save_fpath_img_cr = os.path.join(save_folder_test, 'test01_cropped.png')
# save_fpath_img_cr_rescale = os.path.join(save_folder_test, 'test01_cropped_rescale.png')
# save_fpath_img_orig = os.path.join(save_folder_test, 'test02_orig.png')
# save_fpath_img_cr = os.path.join(save_folder_test, 'test02_cropped.png')
# save_fpath_img_cr_rescale = os.path.join(save_folder_test, 'test02_cropped_rescale.png')
save_fpath_img_orig = os.path.join(save_folder_test, 'test03_orig.png')
save_fpath_img_cr = os.path.join(save_folder_test, 'test03_cropped.png')
save_fpath_img_cr_rescale = os.path.join(save_folder_test, 'test03_cropped_rescale.png')

# proc_type = 'test01'
# proc_type = 'test02'
proc_type = 'test03'


# --------------------------------------------------------------------------------------------------
# load template image and find template in image
# get rgb value of test image
# --------------------------------------------------------------------------------------------------

img_test = cv2.imread(img_test_path)
gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
print(img_test.shape)

# ----------
tmplt = cc_img_bgr_cr.copy()
tmplt_gray = cv2.cvtColor(tmplt, cv2.COLOR_RGB2GRAY)
tmplt_gray_edged = cv2.Canny(tmplt_gray, 50, 200)
(tH, tW) = tmplt_gray_edged.shape[:2]


################
# template matching is not ideal if you are trying to match rotated objects or
# objects that exhibit non-affine transformations.
# If you are concerned with these types of transformations
# you are better of jumping right to keypoint matching.

v_flag = False
found = None

# loop over the scales of the image
for scale in np.linspace(0.75, 1.25, 40)[::-1]:
    print(f'finding template in scale : {scale}')

    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
    r = gray.shape[1] / float(resized.shape[1])

    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break

    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, tmplt_gray_edged, cv2.TM_CCOEFF)

    # takes correlation result and returns a 4-tuple which includes the minimum correlation value,
    # the (x,y)-coordinate of the minimum value, and the (x,y)-coordinate of the maximum value, respectively.
    # We are only interested in the maximum value and (x,y)-coordinate so we keep the maximums and discard the minimums
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    # check to see if the iteration should be visualized
    if v_flag == True:
        # draw a bounding box around the detected region
        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
            (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        cv2.imshow("Visualize", clone)
        cv2.waitKey(0)

    # if we have found a new maximum correlation value, then update
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)


# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# startX = 560
# startY = 615-100
# w = 1620 + 120
# h = 1805 + 300
# endX = startX + w
# endY = startY + h

# CHECK WHETHER THE DETECTION IS CORRECT !!!
# draw a bounding box around the detected result and display the image
img_vis = cv2.rectangle(img_test.copy(), (startX, startY), (endX, endY), (0, 0, 255), 2)
PIL.Image.fromarray(tmplt_gray_edged).show()
PIL.Image.fromarray(cv2.cvtColor(img_vis.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


################
img_test_cr = img_test[startY:endY, startX:endX]
img_test_cr_rescale = cv2.resize(img_test_cr, (int(tW), int(tH)))

PIL.Image.fromarray(cv2.cvtColor(img_test_cr_rescale.astype('uint8'), cv2.COLOR_BGR2RGB)).show()

print(img_test_cr.shape)
print(cc_img_bgr_cr.shape)
print(img_test_cr_rescale.shape)


################
img_vis = img_test_cr_rescale.copy()
color_rect = (255, 0, 0)

val_test = []

for ro in range(n_row):
    for co in range(n_col):
        xmin2 = xmin + sum(shift_x[:co+1])
        ymin2 = ymin + sum(shift_y[:ro+1])
        img_vis = cv2.rectangle(img_vis, (int(xmin2), int(ymin2)), (int(xmin2+rec_width), int(ymin2+rec_height)), color_rect, 3)
        img_tmp = img_test_cr_rescale[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width]
        b, g, r = cv2.split(img_tmp)
        labl, laba, labb = cv2.split(cv2.cvtColor(img_tmp, cv2.COLOR_BGR2Lab))
        hsvh, hsvs, hsvv = cv2.split(cv2.cvtColor(img_tmp, cv2.COLOR_BGR2HSV))
        val_test.append(
            (
                proc_type, 'test',
                np.nan,
                np.nan, np.nan,
                np.nan, np.nan,
                np.nan, np.nan,
                ro, co,
                np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
                np.round(hsvh.mean(), 2), np.round(hsvs.mean(), 2), np.round(hsvv.mean(), 2),
            )
        )

PIL.Image.fromarray(cv2.cvtColor(img_vis.astype('uint8'), cv2.COLOR_BGR2RGB)).show()

# ----------
cv2.imwrite(save_fpath_img_orig, img_test)
cv2.imwrite(save_fpath_img_cr, img_test_cr)
cv2.imwrite(save_fpath_img_cr_rescale, img_test_cr_rescale)

val_test_pd = pd.DataFrame(val_test)
val_test_pd.columns = val_colname
val_test_pd.to_csv(save_fpath_val_test, sep=',', index=False)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# calculate differences metrics: test vs. each proc
# search most similar processed color chart
# --------------------------------------------------------------------------------------------------

def search_best_proc(val_proc, val_test, vari_list, dval_mode='rgb'):
    d_val_list = []
    # wt_r, wt_g, wt_b = 0.3, 0.59, 0.11
    wt_r, wt_g, wt_b = 1.0, 1.0, 1.0
    wt_labl, wt_laba, wt_labb = 1.0, 1.0, 1.0
    wt_hsvh, wt_hsvs, wt_hsvv = 1.0, 1.0, 1.0
    if dval_mode == 'rgb':
        for i, vari in enumerate(vari_list):
            print(f'processing - {vari} - {i} / {len(vari_list)}')
            tmp = val_proc[val_proc.vari == vari]
            d_val = np.sqrt(wt_r * np.sum((val_test.r.values - tmp.r.values) ** 2) + \
                            wt_g * np.sum((val_test.g.values - tmp.g.values) ** 2) + \
                            wt_b * np.sum((val_test.b.values - tmp.b.values) ** 2))
            d_val_list.append(d_val)
    elif dval_mode == 'lab':
        for i, vari in enumerate(vari_list):
            print(f'processing - {vari} - {i} / {len(vari_list)}')
            tmp = val_proc[val_proc.vari == vari]
            d_val = np.sqrt(wt_labl * np.sum((val_test.labl.values - tmp.labl.values) ** 2) + \
                    wt_laba * np.sum((val_test.laba.values - tmp.laba.values) ** 2) + \
                    wt_labb * np.sum((val_test.labb.values - tmp.labb.values) ** 2))
            d_val_list.append(d_val)
    elif dval_mode == 'hsv':
        for i, vari in enumerate(vari_list):
            print(f'processing - {vari} - {i} / {len(vari_list)}')
            tmp = val_proc[val_proc.vari == vari]
            d_val = np.sqrt(wt_hsvh * np.sum((val_test.hsvh.values - tmp.hsvh.values) ** 2) + \
                            wt_hsvs * np.sum((val_test.hsvs.values - tmp.hsvs.values) ** 2) + \
                            wt_hsvv * np.sum((val_test.hsvv.values - tmp.hsvv.values) ** 2))
            d_val_list.append(d_val)
    # ----------
    idx_min = np.argmin(d_val_list)
    vari_min = vari_list[idx_min]
    d_val_min = np.round(d_val_list[idx_min], 2)
    idx_min5 = np.argpartition(d_val_list, 5)[:5]
    vari_min5 = vari_list[idx_min5]
    d_val_min5 = [np.round(d_val_list[idx],2) for idx in idx_min5]
    return idx_min, vari_min, d_val_min, idx_min5, vari_min5, d_val_min5


def search_best_proc_rev(val_proc, val_test, vari_list, n_palette, dval_mode='rgb'):
    d_val_arr = np.zeros(len(vari_list))
    if dval_mode == 'rgb':
        val_proc = np.array(val_proc[['r', 'g', 'b']])
        val_test = np.array(val_test[['r', 'g', 'b']])
    elif dval_mode == 'lab':
        val_proc = np.array(val_proc[['labl', 'laba', 'labb']])
        val_test = np.array(val_test[['labl', 'laba', 'labb']])
    elif dval_mode == 'hsv':
        val_proc = np.array(val_proc[['hsvh', 'hsvs', 'hsvv']])
        val_test = np.array(val_test[['hsvh', 'hsvs', 'hsvv']])
    for i, vari in enumerate(vari_list):
        print(f'processing - {vari} - {i} / {len(vari_list) - 1}')
        d_val_arr[i] = np.sqrt(
            np.sum((val_proc[(i * n_palette):(i * n_palette + n_palette)] - val_test) ** 2))
    # ----------
    idx_min = np.argmin(d_val_arr)
    vari_min = vari_list[idx_min]
    d_val_min = np.round(d_val_arr[idx_min], 2)
    idx_min5 = np.argpartition(d_val_arr, 5)[:5]
    vari_min5 = vari_list[idx_min5]
    d_val_min5 = [np.round(d_val_arr[idx],2) for idx in idx_min5]
    return idx_min, vari_min, d_val_min, idx_min5, vari_min5, d_val_min5


val_ref = pd.read_csv(save_fpath_val_ref)
val_proc = pd.read_csv(save_fpath_val_proc)
val_proc2 = pd.read_csv(save_fpath_val_proc2)
val_proc4 = pd.read_csv(save_fpath_val_proc4)

img_test = cv2.imread(os.path.join(save_folder_test, 'test01_cropped_rescale.png'))
# img_test = cv2.imread(os.path.join(save_folder_test, 'test02_cropped_rescale.png'))
# img_test = cv2.imread(os.path.join(save_folder_test, 'test03_cropped_rescale.png'))

val_test = pd.read_csv(os.path.join(save_folder_val, 'val_test_01.txt'))
# val_test = pd.read_csv(os.path.join(save_folder_val, 'val_test_02.txt'))
# val_test = pd.read_csv(os.path.join(save_folder_val, 'val_test_03.txt'))

# val_proc_all = pd.concat([val_proc, val_ref])
# val_proc_all = pd.concat([val_proc2, val_ref])
val_proc_all = pd.concat([val_proc4, val_ref])
val_proc_all.reset_index(inplace=True)

vari_list = pd.unique(val_proc_all.vari.values)
print(len(vari_list))
print(len(vari_list) * n_col * n_row)
print(len(val_proc_all))

# # tone curve adjustment each by RGB
# idx_min, vari_min, d_val_min, idx_min5, vari_min5, d_val_min5 = \
#     search_best_proc(val_proc=val_proc_all, val_test=val_test, vari_list=vari_list, dval_mode='rgb')
#
# # tone curve adjustment each by LAB
# idx_min2, vari_min2, d_val_min2, idx_min5_2, vari_min5_2, d_val_min5_2 = \
#     search_best_proc(val_proc=val_proc_all, val_test=val_test, vari_list=vari_list, dval_mode='lab')
#
# # tone curve adjustment each by HSV
# idx_min3, vari_min3, d_val_min3, idx_min5_3, vari_min5_3, d_val_min5_3 = \
#     search_best_proc(val_proc=val_proc_all, val_test=val_test, vari_list=vari_list, dval_mode='hsv')

# tone curve adjustment each by gamma + A + B
idx_min4, vari_min4, d_val_min4, idx_min5_4, vari_min5_4, d_val_min5_4 = \
    search_best_proc_rev(val_proc=val_proc_all, val_test=val_test, vari_list=vari_list, n_palette=n_row*n_col, dval_mode='lab')


# print(f'idx min: {idx_min}')
# print(f'vari_min: {vari_min}')
# print(f'd_val_min: {d_val_min}')

print(f'idx min: {idx_min4}')
print(f'vari_min: {vari_min4}')
print(f'd_val_min: {d_val_min4}')

print(f'idx min5: {idx_min5_4}')
print(f'vari_min5: {vari_min5_4}')
print(f'd_val_min5: {d_val_min5_4}')


# t = np.arange(0, len(diff_val), 1)
# plt.plot(t, diff_val)
# plt.show()

# vari = '1.00_0.1_0.9_0.1_0.9_0.1_0.9'
# obj_idx = np.where(np.array(vari_list) == vari)[0]


# ----------
# fpath_match = os.path.join(save_folder_proc, f'color_chart_gamma_tonecurve_proc_{vari_min}.png')
# fpath_match = os.path.join(save_folder_proc2, f'color_chart_gamma_tonecurve_proc2_{vari_min}.png')

# fpath_match2 = os.path.join(save_folder_proc, f'color_chart_gamma_tonecurve_proc_{vari_min2}.png')
# fpath_match2 = os.path.join(save_folder_proc2, f'color_chart_gamma_tonecurve_proc2_{vari_min2}.png')

par_gamma = 1.2
par_labl_l, par_labl_h = 0.4, 0.6
par_laba_l, par_laba_h = 1.0, 0.
par_labb_l, par_labb_h = 1.0, 0.1

img_proc_gamma = adjust_gamma(cc_img_bgr_cr, par_gamma)
img_proc = move_tone_curve(cv2.cvtColor(img_proc_gamma, cv2.COLOR_BGR2Lab),
                           par_labl_l, par_labl_h,
                           par_laba_l, par_laba_h, par_labb_l, par_labb_h)
img_proc = cv2.cvtColor(img_proc, cv2.COLOR_Lab2BGR)

img_to_show = np.hstack([cc_img_bgr_cr, img_proc, img_test])
PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# model transform by spline and process original color chart by learned transformation
# --------------------------------------------------------------------------------------------------

img_ref = cv2.imread(os.path.join(save_folder_ref, 'color_chart_cropped.png'))
w = img_ref.shape[1]
h = img_ref.shape[0]

val_ref = pd.read_csv(save_fpath_val_ref)

# img_test = cv2.imread(os.path.join(save_folder_test, 'test01_cropped_rescale.png'))
# img_test = cv2.imread(os.path.join(save_folder_test, 'test02_cropped_rescale.png'))
img_test = cv2.imread(os.path.join(save_folder_test, 'test03_cropped_rescale.png'))

# val_test = pd.read_csv(os.path.join(save_folder_val, 'val_test_01.txt'))
# val_test = pd.read_csv(os.path.join(save_folder_val, 'val_test_02.txt'))
val_test = pd.read_csv(os.path.join(save_folder_val, 'val_test_03.txt'))

# proc_type = 'test_01'
# proc_type = 'test_02'
proc_type = 'test_03'

val_ref_pd = get_palette_values(img_ref, n_row, n_col, xmin, ymin, shift_x, shift_y, rec_width, rec_height, input_mode='bgr', is_vis=False)
val_test_pd = get_palette_values(img_test, n_row, n_col, xmin, ymin, shift_x, shift_y, rec_width, rec_height, input_mode='bgr', is_vis=False)
print(val_ref_pd.columns)
print(val_test_pd.columns)


# ----------
# B-spline with 4 + 3 - 1 = 6 basis functions
n_knots = 4
degrees = 3
# n_knots = 6
# degrees = 5
model = make_pipeline(SplineTransformer(n_knots=n_knots, degree=degrees), Ridge(alpha=1e-3))


val_list_rgb = [
    [val_ref.r.values, val_ref.g.values, val_ref.b.values],
    [val_test.r.values, val_test.g.values, val_ref.b.values],
    ['r', 'g', 'b']
]

val_list_lab = [
    [val_ref.labl.values, val_ref.laba.values, val_ref.labb.values],
    [val_test.labl.values, val_test.laba.values, val_ref.labb.values],
    ['labl', 'laba', 'labb']
]

val_list_hsv = [
    [val_ref.hsvh.values, val_ref.hsvs.values, val_ref.hsvv.values],
    [val_test.hsvh.values, val_test.hsvs.values, val_ref.hsvv.values],
    ['hsvh', 'hsvs', 'hsvv']
]


for obj_list in [val_list_rgb, val_list_lab, val_list_hsv]:

    if obj_list[2][0] == 'r':
        val0, val1, val2 = cv2.split(img_ref)
        val0 = np.ravel(val0)
        val1 = np.ravel(val1)
        val2 = np.ravel(val2)
        obj_mode = 'rgb'
    elif obj_list[2][0] == 'labl':
        val0, val1, val2 = cv2.split(cv2.cvtColor(img_ref, cv2.COLOR_BGR2Lab))
        val0 = np.ravel(val0)
        val1 = np.ravel(val1)
        val2 = np.ravel(val2)
        obj_mode = 'lab'
    elif obj_list[2][0] == 'hsvh':
        val0, val1, val2 = cv2.split(cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV))
        val0 = np.ravel(val0)
        val1 = np.ravel(val1)
        val2 = np.ravel(val2)
        obj_mode = 'hsv'

    for i in range(len(obj_list[0])):
        x = obj_list[0][i]
        y = obj_list[1][i]
        idx = np.argsort(x)
        x = x[idx][:, np.newaxis]
        y = y[idx]
        model.fit(x, y)
        # ----------
        x_plot = np.linspace(0, 255, 256)
        if obj_mode == 'hsv' and i == 0:
            x_plot = np.linspace(0, 179, 180)
        # ----------
        y_pred = model.predict(x_plot[:, np.newaxis])
        fig, ax = plt.subplots()
        ax.plot(x_plot, y_pred, label="B-spline")
        plt.savefig(os.path.join(base_path, save_folder_result, f'{proc_type}_funcplot_{obj_mode}_{obj_list[2][i]}.png'))
        plt.close()
        if obj_mode == 'rgb':
            if i == 0:
                val2_rev = model.predict(val2[:, np.newaxis]).astype('uint8').reshape(h, w)
            elif i == 1:
                val1_rev = model.predict(val1[:, np.newaxis]).astype('uint8').reshape(h, w)
            elif i == 2:
                val0_rev = model.predict(val0[:, np.newaxis]).astype('uint8').reshape(h, w)
                img_ref_proc = cv2.merge([val0_rev, val1_rev, val2_rev])
                val_proc_pd = get_palette_values(img_ref_proc, n_row, n_col, xmin, ymin, shift_x, shift_y, rec_width, rec_height, input_mode='bgr', is_vis=False)
                comp_values = compare_values(val_ref_pd, val_proc_pd, val_test_pd)
                img_to_save = np.hstack([img_ref, img_ref_proc, img_test])
                fpath_img = os.path.join(save_folder_result, f'{proc_type}_proc_{obj_mode}.png')
                fpath_comp = os.path.join(save_folder_result, f'{proc_type}_proc_{obj_mode}.txt')
                cv2.imwrite(fpath_img, img_to_save)
                comp_values.to_csv(fpath_comp, sep='\t', index=False)
        elif obj_mode == 'lab':
            if i == 0:
                val0_rev = model.predict(val0[:, np.newaxis]).astype('uint8').reshape(h, w)
            elif i == 1:
                val1_rev = model.predict(val1[:, np.newaxis]).astype('uint8').reshape(h, w)
            elif i == 2:
                val2_rev = model.predict(val2[:, np.newaxis]).astype('uint8').reshape(h, w)
                val_proc_pd = get_palette_values(img_ref_proc, n_row, n_col, xmin, ymin, shift_x, shift_y, rec_width, rec_height, input_mode='bgr', is_vis=False)
                comp_values = compare_values(val_ref_pd, val_proc_pd, val_test_pd)
                img_to_save = np.hstack([img_ref, img_ref_proc, img_test])
                fpath_img = os.path.join(save_folder_result, f'{proc_type}_proc_{obj_mode}.png')
                fpath_comp = os.path.join(save_folder_result, f'{proc_type}_proc_{obj_mode}.txt')
                cv2.imwrite(fpath_img, img_to_save)
                comp_values.to_csv(fpath_comp, sep='\t', index=False)
        elif obj_mode == 'hsv':
            if i == 0:
                val0_rev = model.predict(val0[:, np.newaxis]).astype('uint8').reshape(h, w)
            elif i == 1:
                val1_rev = model.predict(val1[:, np.newaxis]).astype('uint8').reshape(h, w)
            elif i == 2:
                val2_rev = model.predict(val2[:, np.newaxis]).astype('uint8').reshape(h, w)
                val_proc_pd = get_palette_values(img_ref_proc, n_row, n_col, xmin, ymin, shift_x, shift_y, rec_width, rec_height, input_mode='bgr', is_vis=False)
                comp_values = compare_values(val_ref_pd, val_proc_pd, val_test_pd)
                img_to_save = np.hstack([img_ref, img_ref_proc, img_test])
                fpath_img = os.path.join(save_folder_result, f'{proc_type}_proc_{obj_mode}.png')
                fpath_comp = os.path.join(save_folder_result, f'{proc_type}_proc_{obj_mode}.txt')
                cv2.imwrite(fpath_img, img_to_save)
                comp_values.to_csv(fpath_comp, sep='\t', index=False)


# --------------------------------------------------------------------------------------------------
# check stats
# --------------------------------------------------------------------------------------------------

ref_rgb_col = ['r_ref', 'g_ref', 'b_ref']
proc_rgb_col = ['r_proc', 'g_proc', 'b_proc']
test_rgb_col = ['r_test', 'g_test', 'b_test']
ref_lab_col = ['labl_ref', 'laba_ref', 'labb_ref']
proc_lab_col = ['labl_proc', 'laba_proc', 'labb_proc']
test_lab_col = ['labl_test', 'laba_test', 'labb_test']
ref_hsv_col = ['hsvh_ref', 'hsvs_ref', 'hsvv_ref']
proc_hsv_col = ['hsvh_proc', 'hsvs_proc', 'hsvv_proc']
test_hsv_col = ['hsvh_test', 'hsvs_test', 'hsvv_test']

dif_rgb_col = ['dif_r_ref', 'dif_r_proc', 'dif_g_ref', 'dif_g_proc', 'dif_b_ref', 'dif_b_proc']
dif_lab_col = ['dif_labl_ref', 'dif_labl_proc', 'dif_laba_ref', 'dif_laba_proc', 'dif_labb_ref', 'dif_labb_proc']
dif_hsv_col = ['dif_hsvh_ref', 'dif_hsvh_proc', 'dif_hsvs_ref', 'dif_hsvs_proc', 'dif_hsvv_ref', 'dif_hsvv_proc']

fpath_comp = os.path.join(save_folder_result, f'test_01_proc_rgb.txt')
comp_values = pd.read_csv(fpath_comp, sep='\t')

obj_col = ['row', 'column'] + dif_rgb_col
print(comp_values[obj_col])

obj_col = ['row', 'column'] + dif_lab_col
print(comp_values[obj_col])

obj_col = ['row', 'column'] + dif_hsv_col
print(comp_values[obj_col])
