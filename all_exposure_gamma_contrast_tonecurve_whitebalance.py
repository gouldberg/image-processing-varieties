
import os
import glob

import numpy as np
import numpy.matlib

import math
import random

import cv2
import PIL.Image


# ----------
# reference:
# https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library


####################################################################################################
# --------------------------------------------------------------------------------------------------
# correct exposure
#  - CLAHE to y in yub
# --------------------------------------------------------------------------------------------------

def correct_exposure(image):
    image= np.copy(image)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    ones= np.ones(img_yuv[...,0].shape)
    ones[img_yuv[...,0]>150]= 0.85
    img_yuv[...,0]= img_yuv[:,:,0]*ones
    # ----------
    img_yuv[...,0] = clahe.apply(img_yuv[...,0])
    img_yuv[...,0] = cv2.equalizeHist(img_yuv[...,0])
    img_yuv[...,0] = clahe.apply(img_yuv[...,0])
    # ----------
    image_res = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    image_res= cv2.fastNlMeansDenoisingColored(image_res,None,3,3,7,21)
    image_res = cv2.cvtColor(image_res, cv2.COLOR_BGR2RGB)
    return image_res


####################################################################################################
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


####################################################################################################
# --------------------------------------------------------------------------------------------------
# adaptive histogram equalization by Y in YUV  (also simple reflection removal)
# --------------------------------------------------------------------------------------------------

def hist_equal_yinyuv(img, clipLimit, tile):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(img_yuv)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile, tile))
    y = clahe.apply(y)
    img_clahe_yuv = cv2.merge([y, u, v])
    img_clahe = cv2.cvtColor(img_clahe_yuv.astype('uint8'), cv2.COLOR_YUV2RGB)
    return img_clahe


####################################################################################################
# --------------------------------------------------------------------------------------------------
# reflection reduction by thresholding and inpainting
# --------------------------------------------------------------------------------------------------

def reflection_reduction_inpaint(img, blur_kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]
    dst_TELEA = cv2.inpaint(img, thresh, 3, cv2.INPAINT_TELEA)
    return dst_TELEA


####################################################################################################
# --------------------------------------------------------------------------------------------------
# white balance corrector
# reference:
# When Color Constancy Goes Wrong: Correcting Improperly White-Balanced Images
# https://github.com/mahmoudnafifi/WB_sRGB
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


def wb_correctImage(I):
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
    mf = sum(np.matlib.repmat(weightsH, 1, 33) * mappingFuncs_rev, 0)  # compute the mapping function
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
# parameters for white balance corrector
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'
params_dir = os.path.join(base_path, '00_white_balance_emulator_params')


# # ----------
# self_features = np.load(os.path.join(params_dir, 'features.npy'))
# # mapping functions to emulate WB effects
# mappingFuncs = np.load(os.path.join(params_dir, 'mappingFuncs.npy'))
# # weight matrix for histogram encoding
# encoderWeights = np.load(os.path.join(params_dir, 'encoderWeights.npy'))
# # bias vector for histogram encoding
# encoderBias = np.load(os.path.join(params_dir, 'encoderBias.npy'))
# K = 25  # K value for nearest neighbor searching


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
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/blue/*')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/yellow/*')))
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/red/*')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.png')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, '*.png')))


# ----------
image_dir = os.path.join(base_path, '00_sample_images/prize/brassband_trading_badge')
img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

image_dir = os.path.join(base_path, '00_sample_images/cushion')
img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

image_dir = os.path.join(base_path, '00_sample_images/cushion/pacman')
img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))


# --------------------------------------------------------------------------------------------------
# select background images to be blended
# --------------------------------------------------------------------------------------------------

image_dir_bg = os.path.join(base_path, '00_sample_images/background')

img_file_list_bg = sorted(glob.glob(os.path.join(image_dir_bg, '*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir_bg, '*.png')))

print(img_file_list_bg)


# --------------------------------------------------------------------------------------------------
# compare various image processing
# --------------------------------------------------------------------------------------------------

# darken
darkness_coeff = 0.5

# brighten
brightness_coeff = 0.5

# adjust brightness and contrast
alpha = 1
beta = 0
beta_by_max = False

# gamma transform
gamma_small = 0.25
gamma_large = 1.75

# move tone curve
low_y = 0.3
high_y = 0.7

# adaptive histogram equalization by CLAHE
clipLimit = 2.0
tile = 8

# reflection reduction inpainting blur kernel
blur_kernel = (3, 3)

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
    # ----------
    # correct exposure
    img_exposure_corrected = correct_exposure(img_rgb)
    # ----------
    # darken, brighten
    img_dark = darken(img_rgb, darkness_coeff)
    img_bright = brighten(img_rgb, brightness_coeff)
    # ----------
    # brightness and contrast adjustment
    img_bc_adjust = brightness_contrast_adjust(
        img=img_rgb, alpha=alpha, beta=beta,
        beta_by_max=beta_by_max)
    # ----------
    # gamma transform
    img_gamma_small = adjust_gamma(img_rgb, gamma_small)
    img_gamma_large = adjust_gamma(img_rgb, gamma_large)
    # ----------
    # move tone curve
    img_tonecurve = move_tone_curve(img_rgb, low_y, high_y)
    # ----------
    # adaptive histogram adaptaion by Y in YUV
    img_clahe = hist_equal_yinyuv(img_rgb, clipLimit, tile)
    # ----------
    # reflection reduction by inpainting
    img_inpaint = reflection_reduction_inpaint(img_rgb, blur_kernel)
    # ----------
    # white balance corrected
    img_wb_corrected = wb_correctImage(img)
    img_wb_corrected_rgb = cv2.cvtColor((img_wb_corrected*255).astype('uint8'), cv2.COLOR_BGR2RGB)
    # ----------
    img_dummy = np.zeros(img_rgb.shape)
    # ----------
    img_to_show0 = np.hstack([
        img_rgb, img_exposure_corrected,
        img_bc_adjust, img_tonecurve])
    img_to_show1 = np.hstack([
        img_clahe, img_inpaint, img_wb_corrected_rgb,
        img_dummy])
    img_to_show2 = np.hstack([
        img_dark, img_gamma_small,
        img_bright, img_gamma_large])
    img_to_show = np.vstack([img_to_show0, img_to_show1, img_to_show2])
    # ----------
    PIL.Image.fromarray(img_to_show.astype('uint8')).show()
    # # ----------
    # h = img_to_show.shape[0]
    # w = img_to_show.shape[1]
    # for bg_idx in range(len(img_file_list_bg)):
    #     img_file_bg = img_file_list_bg[bg_idx]
    #     img_bg = cv2.imread(img_file_bg)
    #     img_bg = cv2.resize(img_bg, (w, h))
    #     img_bg_rgb = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
    #     img_blended = (img_to_show * 0.5 + img_bg_rgb * 0.5).astype('uint8')
    #     PIL.Image.fromarray(img_blended.astype('uint8')).show()
    #
