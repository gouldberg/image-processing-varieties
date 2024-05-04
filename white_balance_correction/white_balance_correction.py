
import os
import glob

import numpy as np
import numpy.matlib

import math
from math import ceil, floor
import random

import cv2
import PIL.Image

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

# reference:
# When Color Constancy Goes Wrong: Correcting Improperly White-Balanced Images
# https://github.com/mahmoudnafifi/WB_sRGB


####################################################################################################
# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------

def normScaling(I, I_corr):
    """ Scales each pixel based on original image energy. """
    norm_I_corr = np.sqrt(np.sum(np.power(I_corr, 2), 1))
    inds = norm_I_corr != 0
    norm_I_corr = norm_I_corr[inds]
    norm_I = np.sqrt(np.sum(np.power(I[inds, :], 2), 1))
    I_corr[inds, :] = I_corr[inds, :] / np.tile(
    norm_I_corr[:, np.newaxis], 3) * np.tile(norm_I[:, np.newaxis], 3)
    return I_corr


def kernelP(rgb):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric
            characterization based on polynomial modeling." Color Research &
            Application, 2001. """
    r, g, b = np.split(rgb, 3, axis=1)
    return np.concatenate(
        [rgb, r * g, r * b, g * b, rgb ** 2, r * g * b, np.ones_like(r)], axis=1)


def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def im2double(im):
    """ Returns a double image [0,1] of the uint8 im [0,255]. """
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# White-Balance Corrector
# --------------------------------------------------------------------------------------------------

def encode(hist):
    """Generates a compacted feature of a given RGB-uv histogram tensor."""
    histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]), (1, int(hist.size / 3)), order="F")
    histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]), (1, int(hist.size / 3)), order="F")
    histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]), (1, int(hist.size / 3)), order="F")
    hist_reshaped = np.append(histR_reshaped, [histG_reshaped, histB_reshaped])
    feature = np.dot(hist_reshaped - encoderBias.transpose(), encoderWeights)
    return feature


def rgb_uv_hist(I):
    """Computes an RGB-uv histogram tensor."""
    sz = np.shape(I)  # get size of current image
    if sz[0] * sz[1] > 202500:  # resize if it is larger than 450*450
        factor = np.sqrt(202500 / (sz[0] * sz[1]))  # rescale factor
        newH = int(np.floor(sz[0] * factor))
        newW = int(np.floor(sz[1] * factor))
        # I = imresize(I, output_shape=(newW, newH))
        I = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)
    I_reshaped = I[(I > 0).all(axis=2)]
    eps = 6.4 / h
    # A = np.arange(-3.2, 3.19, eps)  # dummy vector
    # hist = np.zeros((A.size, A.size, 3))  # histogram will be stored here
    hist = np.zeros((h, h, 3))  # histogram will be stored here
    Iy = np.linalg.norm(I_reshaped, axis=1)  # intensity vector
    for i in range(3):  # for each histogram layer, do
        r = []  # excluded channels will be stored here
        for j in range(3):  # for each color channel do
            if j != i:  # if current channel does not match current layer,
                r.append(j)  # exclude it
        Iu = np.log(I_reshaped[:, i] / I_reshaped[:, r[1]])
        Iv = np.log(I_reshaped[:, i] / I_reshaped[:, r[0]])
        hist[:, :, i], _, _ = np.histogram2d(
            Iu, Iv, bins=h, range=((-3.2 - eps / 2, 3.2 - eps / 2),) * 2, weights=Iy)
        norm_ = hist[:, :, i].sum()
        hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)  # (hist/norm)^(1/2)
    return hist


def correctImage(I):
    """ White balance a given image I. """
    I = I[..., ::-1]  # convert from BGR to RGB
    I = im2double(I)  # convert to double
    # Convert I to float32 may speed up the process.
    feature = encode(rgb_uv_hist(I))
    # Do
    # ```python
    # feature_diff = self.features - feature
    # D_sq = np.einsum('ij,ij->i', feature_diff, feature_diff)[:, None]
    # ```
    D_sq = np.einsum('ij, ij ->i', self_features, self_features)[:, None] + \
        np.einsum('ij, ij ->i', feature, feature) - 2 * self_features.dot(feature.T)
    # ----------
    # get smallest K distances
    idH = D_sq.argpartition(K, axis=0)[:K]
    mappingFuncs_rev = np.squeeze(mappingFuncs[idH, :])
    dH = np.sqrt(np.take_along_axis(D_sq, idH, axis=0))
    weightsH = np.exp(-(np.power(dH, 2)) / (2 * np.power(sigma, 2)))  # compute weights
    weightsH = weightsH / sum(weightsH)  # normalize blending weights
    mf = sum(numpy.matlib.repmat(weightsH, 1, 33) * mappingFuncs_rev, 0)  # compute the mapping function
    mf = mf.reshape(11, 3, order="F")  # reshape it to be 9 * 3
    I_corr = colorCorrection(I, mf)  # apply it!
    return I_corr


def colorCorrection(input, m):
    """ Applies a mapping function m to a given input image. """
    sz = np.shape(input)  # get size of input image
    I_reshaped = np.reshape(input, (int(input.size / 3), 3), order="F")
    kernel_out = kernelP(I_reshaped)
    out = np.dot(kernel_out, m)
    if gamut_mapping == 1:
        # scaling based on input image energy
        out = normScaling(I_reshaped, out)
    elif gamut_mapping == 2:
        # clip out-of-gamut pixels
        out = outOfGamutClipping(out)
    else:
        raise Exception('Wrong gamut_mapping value')
    # reshape output image back to the original image shape
    out = out.reshape(sz[0], sz[1], sz[2], order="F")
    out = out.astype('float32')[..., ::-1]  # convert from BGR to RGB
    return out


####################################################################################################
# --------------------------------------------------------------------------------------------------
# adjust white balance: very simple version
#   - good for yellow but not good for blue
# --------------------------------------------------------------------------------------------------

def adjust_white_balance(image: np.ndarray) -> np.ndarray:
    # white balance adjustment for strong neutral white
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    avg_a = np.average(image[:, :, 1])
    avg_b = np.average(image[:, :, 2])
    image[:, :, 1] = image[:, :, 1] - (
        (avg_a - 128) * (image[:, :, 0] / 255.0) * 1.1
    )
    image[:, :, 2] = image[:, :, 2] - (
        (avg_b - 128) * (image[:, :, 0] / 255.0) * 1.1
    )
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image


####################################################################################################
# --------------------------------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

params_dir = os.path.join(base_path, '00_white_balance_emulator_params')


# ----------
self_features = np.load(os.path.join(params_dir, 'features.npy'))

# mapping functions to emulate WB effects
mappingFuncs = np.load(os.path.join(params_dir, 'mappingFuncs.npy'))

# weight matrix for histogram encoding
encoderWeights = np.load(os.path.join(params_dir, 'encoderWeights.npy'))

# bias vector for histogram encoding
encoderBias = np.load(os.path.join(params_dir, 'encoderBias.npy'))

K = 25  # K value for nearest neighbor searching


# ----------
# HERE IS UPGRADED VERSION
# ----------
self_features = np.load(os.path.join(params_dir, 'features+.npy'))

# mapping functions to emulate WB effects
mappingFuncs = np.load(os.path.join(params_dir, 'mappingFuncs+.npy'))

# weight matrix for histogram encoding
encoderWeights = np.load(os.path.join(params_dir, 'encoderWeights+.npy'))

# bias vector for histogram encoding
encoderBias = np.load(os.path.join(params_dir, 'encoderBias+.npy'))

K = 75  # K value for nearest neighbor searching



# ----------
h = 60  # histogram bin width

sigma = 0.25  # fall off factor for KNN


# options: 1 scaling, 2 clipping
# 1: scaling based on input image energy
# 2: clip out-of-gamut pixels
# If the image is over-saturated, scaling (1) is recommended.
gamut_mapping = 1


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

image_dir = os.path.join(base_path, '00_sample_images')

# image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
# print(f'num of images: {len(image_path_list)}')


# ----------
# almost no change
# img_file = os.path.join(image_dir, 'road_day.png')
# img_file = os.path.join(image_dir, 'horseshoe_bend.png')
# img_file = os.path.join(image_dir, 'bedroom.jpg')
# img_file = os.path.join(image_dir, 'strong_light.jpg')


# this is interesting
img_file = os.path.join(image_dir, 'white_balance/yellow/bathroom_with_light_001.jpg')
img_file = os.path.join(image_dir, 'white_balance/yellow/bathroom_with_light_002.jpg')
img_file = os.path.join(image_dir, 'white_balance/yellow/bird_yellow.jpg')
img_file = os.path.join(image_dir, 'white_balance/yellow/living_room.jpg')

# sun glare case
# img_file = os.path.join(image_dir, 'sun_flare_light_glare/driving_sun_glare.jpg')
# img_file = os.path.join(image_dir, 'sun_flare_light_glare/girl_with_glare_light.png')


# ----------
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/blue/*')))
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/yellow/*')))
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/red/*')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.png')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, '*.png')))


# --------------------------------------------------------------------------------------------------
# correct white balance
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
    # correct white balance
    # --------------------------------------------------------------------------------------------------
    # ----------
    # corrector (bgr input)
    img_corrected1 = correctImage(img)
    img_corrected_rgb1 = cv2.cvtColor((img_corrected1*255).astype('uint8'), cv2.COLOR_BGR2RGB)
    # ----------
    # simple version (rgb input)
    img_corrected_rgb2 = adjust_white_balance(img_rgb)
    # ----------
    img_to_show = np.hstack([img_rgb, img_corrected_rgb1, img_corrected_rgb2])
    PIL.Image.fromarray(img_to_show.astype('uint8')).show()
