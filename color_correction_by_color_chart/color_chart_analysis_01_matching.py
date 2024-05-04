
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

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copy import deepcopy


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

# def move_tone_curve(img, low_y, high_y):
#     """Rescales the relationship between bright and dark areas of the image by manipulating its tone curve.
#     Args:
#         img (numpy.ndarray): RGB or grayscale image.
#         low_y (float): y-position of a Bezier control point used
#             to adjust the tone curve, must be in range [0, 1]
#         high_y (float): y-position of a Bezier control point used
#             to adjust image tone curve, must be in range [0, 1]
#     """
#     input_dtype = img.dtype
#     # ----------
#     if low_y < 0 or low_y > 1:
#         raise ValueError("low_shift must be in range [0, 1]")
#     if high_y < 0 or high_y > 1:
#         raise ValueError("high_shift must be in range [0, 1]")
#     # ----------
#     if input_dtype != np.uint8:
#         raise ValueError("Unsupported image type {}".format(input_dtype))
#     # ----------
#     t = np.linspace(0.0, 1.0, 256)
#     # ----------
#     # Defines responze of a four-point bezier curve
#     def evaluate_bez(t):
#         return 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t**2 * high_y + t**3
#     # ----------
#     evaluate_bez = np.vectorize(evaluate_bez)
#     remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)
#     # ----------
#     return cv2.LUT(img, lut=remapping)


def move_tone_curve(img, low_0, high_0, low_1, high_1, low_2, high_2):
    val0, val1, val2 = cv2.split(img)
    t = np.linspace(0.0, 1.0, 256)
    def evaluate_bez_0(t):
        return 3 * (1 - t) ** 2 * t * low_0 + 3 * (1 - t) * t**2 * high_0 + t**3
    def evaluate_bez_1(t):
        return 3 * (1 - t) ** 2 * t * low_1 + 3 * (1 - t) * t**2 * high_1 + t**3
    def evaluate_bez_2(t):
        return 3 * (1 - t) ** 2 * t * low_2 + 3 * (1 - t) * t**2 * high_2 + t**3
    # ----------
    evaluate_bez_0 = np.vectorize(evaluate_bez_0)
    evaluate_bez_1 = np.vectorize(evaluate_bez_1)
    evaluate_bez_2 = np.vectorize(evaluate_bez_2)
    remapping_0 = np.rint(evaluate_bez_0(t) * 255).astype(np.uint8)
    remapping_1 = np.rint(evaluate_bez_1(t) * 255).astype(np.uint8)
    remapping_2 = np.rint(evaluate_bez_2(t) * 255).astype(np.uint8)
    # ----------
    val0_remap = cv2.LUT(val0, lut=remapping_0)
    val1_remap = cv2.LUT(val1, lut=remapping_1)
    val2_remap = cv2.LUT(val2, lut=remapping_2)
    return cv2.merge([val0_remap, val1_remap, val2_remap])


####################################################################################################
# --------------------------------------------------------------------------------------------------
# histogram matching
# --------------------------------------------------------------------------------------------------

def histogram_matching(src, ref):
    src_lookup = src.reshape(-1)
    src_counts = np.bincount(src_lookup)
    tmpl_counts = np.bincount(ref.reshape(-1))
    # ----------
    # omit values where the count was 0
    tmpl_values = np.nonzero(tmpl_counts)[0]
    tmpl_counts = tmpl_counts[tmpl_values]
    # ----------
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / src.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / ref.size
    # ----------
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(src.shape)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# pixel domain adaptation
# --------------------------------------------------------------------------------------------------

# class TransformerInterface(Protocol):
#     @abc.abstractmethod
#     def inverse_transform(self, X: np.ndarray) -> np.ndarray:
#         ...
#     @abc.abstractmethod
#     def fit(self, X: np.ndarray, y=None):
#         ...
#     @abc.abstractmethod
#     def transform(self, X: np.ndarray, y=None) -> np.ndarray:
#         ...

class DomainAdapter:
    def __init__(self, transformer, ref_img, color_conversions=(None, None)):
        self.color_in, self.color_out = color_conversions
        self.source_transformer = deepcopy(transformer)
        self.target_transformer = transformer
        self.target_transformer.fit(self.flatten(ref_img))

    # ----------
    def to_colorspace(self, img):
        if self.color_in is None:
            return img
        return cv2.cvtColor(img, self.color_in)

    # ----------
    def from_colorspace(self, img):
        if self.color_out is None:
            return img
        return cv2.cvtColor(img.astype('uint8'), self.color_out)

    # ----------
    def flatten(self, img):
        img = self.to_colorspace(img)
        img = img.astype('float32') / 255.
        return img.reshape(-1, 3)

    # ----------
    def reconstruct(self, pixels, h, w):
        pixels = (np.clip(pixels, 0, 1) * 255).astype('uint8')
        return self.from_colorspace(pixels.reshape(h, w, 3))

    # ----------
    @staticmethod
    def _pca_sign(x):
        return np.sign(np.trace(x.components_))

    # ----------
    def __call__(self, image: np.ndarray):
        h, w, _ = image.shape
        pixels = self.flatten(image)
        self.source_transformer.fit(pixels)
        # ----------
        if self.target_transformer.__class__ in (PCA,):
            # dirty hack to make sure colors are not inverted
            if self._pca_sign(self.target_transformer) != self._pca_sign(self.source_transformer):
                self.target_transformer.components_ *= -1
        # ----------
        representation = self.source_transformer.transform(pixels)
        result = self.target_transformer.inverse_transform(representation)
        return self.reconstruct(result, h, w)


def adapt_pixel_distribution(img, ref, transform_type='pca', weight=0.5):
    initial_type = img.dtype
    transformer = {"pca": PCA, "standard": StandardScaler, "minmax": MinMaxScaler}[transform_type]()
    adapter = DomainAdapter(transformer=transformer, ref_img=ref)
    result = adapter(img).astype("float32")
    blended = (img.astype("float32") * (1 - weight) + result * weight).astype(initial_type)
    return blended


def adapt_pixel_distribution_simple_pca(src_img, tgt_img, weight=0.5):
    initial_type = src_img.dtype
    tgt_pixels = tgt_img.astype('float32') / 255.
    tgt_pixels = tgt_pixels.reshape(-1, 3)
    # ----------
    tgt_mean = np.nanmean(tgt_pixels, axis=0)
    tgt_scale = np.nanstd(tgt_pixels, axis=0)
    tgt_S = np.cov((tgt_pixels - tgt_mean).T, bias=1)
    # tgt_S = np.cov(((tgt_pixels - tgt_mean)/tgt_scale).T, bias=1)
    # ----------
    # tgt_eig = np.linalg.eig(tgt_S)[0]
    tgt_eigvec = np.linalg.eig(tgt_S)[1]
    # tgt_idx = np.argsort(tgt_eig)[::-1]
    # tgt_eig = tgt_eig[tgt_idx]
    # tgt_eigvec = tgt_eigvec[tgt_idx]
    # ----------
    h, w, _ = src_img.shape
    src_pixels = src_img.astype('float32') / 255.
    src_pixels = src_pixels.reshape(-1, 3)
    # ----------
    src_mean = np.mean(src_pixels, axis=0)
    src_scale = np.nanstd(src_pixels, axis=0)
    src_S = np.cov((src_pixels - src_mean).T, bias=1)
    # src_S = np.cov(((src_pixels - src_mean)/src_scale).T, bias=1)
    # ----------
    # src_eig = np.linalg.eig(src_S)[0]
    src_eigvec = np.linalg.eig(src_S)[1]
    # src_idx = np.argsort(src_eig)[::-1]
    # src_eig = src_eig[src_idx]
    # src_eigvec = src_eigvec[src_idx]
    # ----------
    if np.sign(np.trace(src_eigvec)) != np.sign(np.trace(tgt_eigvec)):
        tgt_eigvec = -1 * tgt_eigvec
    representation = np.dot(src_pixels - src_mean, src_eigvec.T)
    result = np.dot(representation, tgt_eigvec) + tgt_mean
    pixels = (np.clip(result, 0, 1) * 255).astype('uint8')
    result = pixels.reshape(h, w, 3).astype('float32')
    src_img_transformed = (src_img.astype("float32") * (1 - weight) + result * weight).astype(initial_type)
    return src_img_transformed


# Vt: components_
def adapt_pixel_distribution_simple_pca2(src_img, tgt_img, weight=0.5):
    initial_type = src_img.dtype
    tgt_pixels = tgt_img.astype('float32') / 255.
    tgt_pixels = tgt_pixels.reshape(-1, 3)
    # ----------
    tgt_mean = np.mean(tgt_pixels, axis=0)
    tgt_scale = np.nanstd(tgt_pixels, axis=0)
    tgt_pixels -= tgt_mean
    # tgt_pixels /= tgt_scale
    tU, tS, tVt = np.linalg.svd(tgt_pixels, full_matrices=False)
    # t_max_abs_cols = np.argmax(np.abs(tU), axis=0)
    # t_signs = np.sign(tU[t_max_abs_cols, range(tU.shape[1])])
    # tU *= t_signs
    # tVt *= t_signs[:, np.newaxis]
    # ----------
    h, w, _ = src_img.shape
    src_pixels = src_img.astype('float32') / 255.
    src_pixels = src_pixels.reshape(-1, 3)
    # ----------
    src_mean = np.mean(src_pixels, axis=0)
    src_scale = np.nanstd(src_pixels, axis=0)
    src_pixels -= src_mean
    # src_pixels /= src_scale
    sU, sS, sVt = np.linalg.svd(src_pixels, full_matrices=False)
    # s_max_abs_cols = np.argmax(np.abs(sU), axis=0)
    # s_signs = np.sign(sU[s_max_abs_cols, range(sU.shape[1])])
    # sU *= s_signs
    # sVt *= s_signs[:, np.newaxis]
    # ----------
    if np.sign(np.trace(sVt)) != np.sign(np.trace(tVt)):
        tVt = -1 * tVt
    # ----------
    representation = np.dot(src_pixels, sVt.T)
    result = np.dot(representation, tVt) + tgt_mean
    pixels = (np.clip(result, 0, 1) * 255).astype('uint8')
    result = pixels.reshape(h, w, 3).astype('float32')
    src_img_transformed = (src_img.astype("float32") * (1 - weight) + result * weight).astype(initial_type)
    return src_img_transformed


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

save_folder_ref = os.path.join(save_folder, 'color_chart_ref')
save_folder_proc = os.path.join(save_folder, 'color_chart_proc')
save_folder_proc2 = os.path.join(save_folder, 'color_chart_proc2')
save_folder_test = os.path.join(save_folder, 'color_chart_test')
save_folder_val = os.path.join(save_folder, 'val')

# if os.path.exists(save_folder_proc):
#     shutil.rmtree(save_folder_proc)
#
# if not os.path.exists(save_folder_proc):
#     os.makedirs(save_folder_proc, exist_ok=True)

# ----------
save_fpath_val_ref = os.path.join(save_folder_val, 'val_ref.txt')
save_fpath_val_proc = os.path.join(save_folder_val, f'val_gamma_tonecurve_proc.txt')
save_fpath_val_proc2 = os.path.join(save_folder_val, f'val_gamma_tonecurve_proc2.txt')


# ----------
val_colname = [
    'type', 'vari', 'gamma',
    'tone_0_low', 'tone_0_high',
    'tone_1_low', 'tone_1_high',
    'tone_2_low', 'tone_2_high',
    'column', 'row',
    'r', 'g', 'b',
    'labl', 'laba', 'labb'
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
# get RGB value of each color
# --------------------------------------------------------------------------------------------------

xmin, ymin = 0, 0
width_ref = cc_img_bgr_cr.shape[1]
height_ref = cc_img_bgr_cr.shape[0]

n_col, n_row = 9, 8

shift_x = [70+200] + [200] * (n_row - 1)
shift_y = [70+180] + [200] * (n_col - 1)
rec_width = 60
rec_height = rec_width

img_vis = cc_img_bgr_cr.copy()
color_rect = (255, 0, 0)

val_ref = []
r_val_ref, g_val_ref, b_val_ref = [], [], []
labl_val_ref, laba_val_ref, labb_val_ref = [], [], []

for co in range(n_col):
    for ro in range(n_row):
        xmin2 = xmin + sum(shift_x[:ro+1])
        ymin2 = ymin + sum(shift_y[:co+1])
        img_vis = cv2.rectangle(img_vis, (int(xmin2), int(ymin2)), (int(xmin2+rec_width), int(ymin2+rec_height)), color_rect, 3)
        b, g, r = cv2.split(cc_img_bgr_cr[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width])
        labl, laba, labb = cv2.split(cv2.cvtColor(cc_img_bgr_cr[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width], cv2.COLOR_BGR2LAB))
        r_val_ref.append(r)
        g_val_ref.append(g)
        b_val_ref.append(b)
        labl_val_ref.append(labl)
        laba_val_ref.append(laba)
        labb_val_ref.append(labb)
        val_ref.append(('ref', 'ref',
                        np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan,
                        co, ro,
                        np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                        np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
                        ))

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
if os.path.exists(save_folder_proc):
    shutil.rmtree(save_folder_proc)
    os.makedirs(save_folder_proc)

val_proc = []

for par_gamma in par_gamma_list:
    img_proc_gamma = adjust_gamma(cv2.cvtColor(cc_img_bgr_cr, cv2.COLOR_BGR2RGB), par_gamma)
    for par_r in par_r_list:
        for par_g in par_g_list:
            for par_b in par_b_list:
                # if par_r == par_g and par_r == par_b:
                print(f"gamma: {format(par_gamma, '.2f')}  r: ({format(par_r[0], '.1f')}, {format(par_r[1], '.1f')})  g: ({format(par_g[0], '.1f')}, {format(par_g[1], '.1f')})  b: ({format(par_b[0], '.1f')}, {format(par_b[1], '.1f')})")
                img_proc = move_tone_curve(img_proc_gamma, par_r[0], par_r[1], par_g[0], par_g[1], par_b[0], par_b[1])
                vari = f"{format(par_gamma, '.2f')}_{format(par_r[0], '.1f')}_{format(par_r[1], '.1f')}_{format(par_g[0], '.1f')}_{format(par_g[1], '.1f')}_{format(par_b[0], '.1f')}_{format(par_b[1], '.1f')}"
                fpath = os.path.join(save_folder_proc, f'color_chart_gamma_tonecurve_proc_{vari}.png')
                cv2.imwrite(fpath, cv2.cvtColor(img_proc, cv2.COLOR_RGB2BGR))
                for co in range(n_col):
                    for ro in range(n_row):
                        xmin2 = xmin + sum(shift_x[:ro + 1])
                        ymin2 = ymin + sum(shift_y[:co + 1])
                        r, g, b = cv2.split(img_proc[ymin2:ymin2 + rec_height, xmin2:xmin2 + rec_width])
                        labl, laba, labb = cv2.split(cv2.cvtColor(img_proc[ymin2:ymin2 + rec_height, xmin2:xmin2 + rec_width], cv2.COLOR_RGB2Lab))
                        val_proc.append(('gamma_tonecurve_proc', vari,
                                        np.round(par_gamma, 2),
                                        np.round(par_r[0], 1), np.round(par_r[1], 1),
                                        np.round(par_g[0], 1), np.round(par_g[1], 1),
                                        np.round(par_b[0], 1), np.round(par_b[1], 1),
                                        co, ro,
                                        np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                                        np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2)))

rgb_val_proc_pd = pd.DataFrame(val_proc)
rgb_val_proc_pd.columns = val_colname
rgb_val_proc_pd.to_csv(save_fpath_val_proc, sep=',', index=False)


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
if os.path.exists(save_folder_proc2):
    shutil.rmtree(save_folder_proc2)
    os.makedirs(save_folder_proc2)

val_proc2 = []

for par_gamma in par_gamma_list:
    img_proc_gamma = adjust_gamma(cv2.cvtColor(cc_img_bgr_cr, cv2.COLOR_BGR2RGB), par_gamma)
    for par_labl in par_labl_list:
        for par_laba in par_laba_list:
            for par_labb in par_labb_list:
                # if par_labl == par_laba and par_laba == par_labb:
                print(f"gamma: {format(par_gamma, '.2f')}  l: ({format(par_labl[0], '.1f')}, {format(par_labl[1], '.1f')})  a: ({format(par_laba[0], '.1f')}, {format(par_laba[1], '.1f')})  labb: ({format(par_labb[0], '.1f')}, {format(par_labb[1], '.1f')})")
                img_proc = move_tone_curve(cv2.cvtColor(img_proc_gamma, cv2.COLOR_RGB2Lab), par_labl[0], par_labl[1], par_laba[0], par_laba[1], par_labb[0], par_labb[1])
                vari = f"{format(par_gamma, '.2f')}_{format(par_labl[0], '.1f')}_{format(par_labl[1], '.1f')}_{format(par_laba[0], '.1f')}_{format(par_laba[1], '.1f')}_{format(par_labb[0], '.1f')}_{format(par_labb[1], '.1f')}"
                fpath = os.path.join(save_folder_proc2, f'color_chart_gamma_tonecurve_proc2_{vari}.png')
                cv2.imwrite(fpath, cv2.cvtColor(img_proc, cv2.COLOR_Lab2BGR))
                for co in range(n_col):
                    for ro in range(n_row):
                        xmin2 = xmin + sum(shift_x[:ro + 1])
                        ymin2 = ymin + sum(shift_y[:co + 1])
                        r, g, b = cv2.split(cv2.cvtColor(img_proc[ymin2:ymin2 + rec_height, xmin2:xmin2 + rec_width], cv2.COLOR_Lab2RGB))
                        labl, laba, labb = cv2.split(img_proc[ymin2:ymin2 + rec_height, xmin2:xmin2 + rec_width])
                        val_proc2.append(('gamma_tonecurve_proc', vari,
                                        np.round(par_gamma, 2),
                                        np.round(par_r[0], 1), np.round(par_r[1], 1),
                                        np.round(par_g[0], 1), np.round(par_g[1], 1),
                                        np.round(par_b[0], 1), np.round(par_b[1], 1),
                                        co, ro,
                                        np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                                        np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2)))

rgb_val_proc2_pd = pd.DataFrame(val_proc2)
rgb_val_proc2_pd.columns = val_colname
rgb_val_proc2_pd.to_csv(save_fpath_val_proc2, sep=',', index=False)


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
r_val_test, g_val_test, b_val_test = [], [], []
labl_val_test, laba_val_test, labb_val_test = [], [], []

for co in range(n_col):
    for ro in range(n_row):
        xmin2 = xmin + sum(shift_x[:ro+1])
        ymin2 = ymin + sum(shift_y[:co+1])
        img_vis = cv2.rectangle(img_vis, (int(xmin2), int(ymin2)), (int(xmin2+rec_width), int(ymin2+rec_height)), color_rect, 3)
        b, g, r = cv2.split(img_test_cr_rescale[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width])
        labl, laba, labb = cv2.split(cv2.cvtColor(img_test_cr_rescale[ymin2:ymin2+rec_height, xmin2:xmin2+rec_width], cv2.COLOR_BGR2Lab))
        r_val_test.append(r)
        g_val_test.append(g)
        b_val_test.append(b)
        labl_val_test.append(labl)
        laba_val_test.append(laba)
        labb_val_test.append(labb)
        val_test.append(
            (
                proc_type, 'test',
                np.nan,
                np.nan, np.nan,
                np.nan, np.nan,
                np.nan, np.nan,
                co, ro,
                np.round(r.mean(), 2), np.round(g.mean(), 2), np.round(b.mean(), 2),
                np.round(labl.mean(), 2), np.round(laba.mean(), 2), np.round(labb.mean(), 2),
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

def find_similar_proc(val_proc, val_test, vari_list, dval_mode='rgb'):
    d_val_list = []
    for vari in vari_list:
        tmp = val_proc[val_proc.vari == vari]
        # ----------
        if dval_mode == 'rgb':
            wt_r, wt_g, wt_b = 0.3, 0.59, 0.11
            # weighed euclidean distance
            d_val = np.sqrt(wt_r * np.sum((val_test.r.values - tmp.r.values) ** 2) + \
                    wt_g * np.sum((val_test.g.values - tmp.g.values) ** 2) + \
                    wt_b * np.sum((val_test.b.values - tmp.b.values) ** 2))
        elif dval_mode == 'lab':
            wt_labl, wt_laba, wt_labb = 1.0, 1.0, 1.0
            d_val = np.sqrt(wt_labl * np.sum((val_test.labl.values - tmp.labl.values) ** 2) + \
                    wt_laba * np.sum((val_test.laba.values - tmp.laba.values) ** 2) + \
                    wt_labb * np.sum((val_test.labb.values - tmp.labb.values) ** 2))
        d_val_list.append(d_val)
        # ----------
    idx_min = np.argmin(d_val_list)
    vari_min = vari_list[idx_min]
    d_val_min = np.round(d_val_list[idx_min], 2)
    idx_min5 = np.argpartition(d_val_list, 5)[:5]
    vari_min5 = vari_list[idx_min5]
    d_val_min5 = [np.round(d_val_list[idx],2) for idx in idx_min5]
    return idx_min, vari_min, d_val_min, idx_min5, vari_min5, d_val_min5


val_ref = pd.read_csv(save_fpath_val_ref)
val_proc = pd.read_csv(save_fpath_val_proc)
val_proc2 = pd.read_csv(save_fpath_val_proc2)

# img_test = cv2.imread(os.path.join(save_folder_test, 'test01_cropped_rescale.png'))
# img_test = cv2.imread(os.path.join(save_folder_test, 'test02_cropped_rescale.png'))
img_test = cv2.imread(os.path.join(save_folder_test, 'test03_cropped_rescale.png'))

# val_test = pd.read_csv(os.path.join(save_folder_val, 'val_test_01.txt'))
# val_test = pd.read_csv(os.path.join(save_folder_val, 'val_test_02.txt'))
val_test = pd.read_csv(os.path.join(save_folder_val, 'val_test_03.txt'))

# val_proc_all = pd.concat([val_proc, val_ref])
val_proc_all = pd.concat([val_proc2, val_ref])

vari_list = pd.unique(val_proc_all.vari.values)

# tone curve adjustment each by RGB
idx_min, vari_min, d_val_min, idx_min5, vari_min5, d_val_min5 = \
    find_similar_proc(val_proc=val_proc_all, val_test=val_test, vari_list=vari_list, dval_mode='rgb')

# tone curve adjustment each by LAB
idx_min2, vari_min2, d_val_min2, idx_min5_2, vari_min5_2, d_val_min5_2 = \
    find_similar_proc(val_proc=val_proc_all, val_test=val_test, vari_list=vari_list, dval_mode='lab')

print(f'idx min: {idx_min}')
print(f'vari_min: {vari_min}')
print(f'd_val_min: {d_val_min}')
print(f'idx min: {idx_min2}')
print(f'vari_min: {vari_min2}')
print(f'd_val_min: {d_val_min2}')

print(f'idx min5: {idx_min5}')
print(f'vari_min5: {vari_min5}')
print(f'd_val_min5: {d_val_min5}')
print(f'idx min5: {idx_min5_2}')
print(f'vari_min5: {vari_min5_2}')
print(f'd_val_min5: {d_val_min5_2}')


# t = np.arange(0, len(diff_val), 1)
# plt.plot(t, diff_val)
# plt.show()

# vari = '1.00_0.1_0.9_0.1_0.9_0.1_0.9'
# obj_idx = np.where(np.array(vari_list) == vari)[0]


# ----------
# fpath_match = os.path.join(save_folder_proc, f'color_chart_gamma_tonecurve_proc_{vari_min}.png')
fpath_match = os.path.join(save_folder_proc2, f'color_chart_gamma_tonecurve_proc2_{vari_min}.png')

# fpath_match2 = os.path.join(save_folder_proc, f'color_chart_gamma_tonecurve_proc_{vari_min2}.png')
fpath_match2 = os.path.join(save_folder_proc2, f'color_chart_gamma_tonecurve_proc2_{vari_min2}.png')

img_match = cv2.imread(fpath_match)
img_match2 = cv2.imread(fpath_match2)

img_to_show0 = np.hstack([cc_img_bgr_cr, img_test])
img_to_show1 = np.hstack([img_match, img_match2])
img_to_show = np.vstack([img_to_show0, img_to_show1])
PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# histogram matching:
# change src image (color chart) by referencing to test image
# --------------------------------------------------------------------------------------------------

# img_test = cv2.imread(os.path.join(save_folder_test, 'test01_cropped_rescale.png'))
# img_test = cv2.imread(os.path.join(save_folder_test, 'test02_cropped_rescale.png'))
img_test = cv2.imread(os.path.join(save_folder_test, 'test03_cropped_rescale.png'))

src = cc_img_bgr_cr.copy()
ref = img_test.copy()

matched = np.zeros(src.shape)
matched[..., 0] = histogram_matching(src[..., 0], ref[..., 0])
matched[..., 1] = histogram_matching(src[..., 1], ref[..., 1])
matched[..., 2] = histogram_matching(src[..., 2], ref[..., 2])


img_to_show = np.hstack([src, ref, matched])
PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# pixel domain adaptation
# --------------------------------------------------------------------------------------------------

# img_test = cv2.imread(os.path.join(save_folder_test, 'test01_cropped_rescale.png'))
# img_test = cv2.imread(os.path.join(save_folder_test, 'test02_cropped_rescale.png'))
img_test = cv2.imread(os.path.join(save_folder_test, 'test03_cropped_rescale.png'))

src = cc_img_bgr_cr.copy()
tgt = img_test.copy()


# ----------
# transform type is 'pca', 'standard' or 'minxmax'

transform_type = 'pca'
# transform_type = 'standard'
# transform_type = 'minmax'

weight = 0.5

src_img_transformed = adapt_pixel_distribution(
    img=src,
    ref=tgt,
    transform_type=transform_type,
    weight=weight
)


src_img_transformed2 = adapt_pixel_distribution_simple_pca(
    src_img=src,
    tgt_img=tgt,
    weight=weight
)

src_img_transformed3 = adapt_pixel_distribution_simple_pca2(
    src_img=src,
    tgt_img=tgt,
    weight=weight
)

img_to_show = np.hstack([src, tgt, src_img_transformed, src_img_transformed2, src_img_transformed3])
PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


