
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
# White-Balance Emulator for Color Augmentation:
# https://github.com/mahmoudnafifi/WB_color_augmenter/tree/124b62b4ab864fdd3ff371b2e68594a4cf8c9c1e


####################################################################################################
# --------------------------------------------------------------------------------------------------
# image resizer
# --------------------------------------------------------------------------------------------------

def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method is 'bicubic':
        kernel = cubic
    elif method is 'bilinear':
        kernel = triangle
    else:
        print ('Error: Unidentified method supplied')
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print ('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------

def kernelP9(I):
  """Kernel function: kernel(r, g, b) -> (r, g, b, r2, g2, b2, rg, rb, gb)"""
  return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 0],
                        I[:, 1] * I[:, 1], I[:, 2] * I[:, 2], I[:, 0] * I[:, 1],
                        I[:, 0] * I[:, 2], I[:, 1] * I[:, 2])))


def outOfGamutClipping(I):
  """Clips out-of-gamut pixels."""
  I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
  I[I < 0] = 0  # any pixel is below 0, clip it to 0
  return I


def changeWB(input, m):
    """Applies a mapping function m to a given input image."""
    sz = np.shape(input)  # get size of input image
    I_reshaped = np.reshape(input, (int(input.size / 3), 3), order="F")
    kernel_out = kernelP9(I_reshaped)  # raise input image to a higher-dim space
    # apply m to the input image after raising it the selected higher degree
    out = np.dot(kernel_out, m)
    out = outOfGamutClipping(out)  # clip out-of-gamut pixels
    # reshape output image back to the original image shape
    out = out.reshape(sz[0], sz[1], sz[2], order="F")
    out = PIL.Image.fromarray(np.uint8(out * 255))
    return out


####################################################################################################
# --------------------------------------------------------------------------------------------------
# White-Balance Emulator
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
        I = imresize(I, output_shape=(newW, newH))
    I_reshaped = I[(I > 0).all(axis=2)]
    eps = 6.4 / h
    A = np.arange(-3.2, 3.19, eps)  # dummy vector
    hist = np.zeros((A.size, A.size, 3))  # histogram will be stored here
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


def generateWbsRGB(I, outNum=10):
    """Generates outNum new images of a given image I."""
    assert (outNum <= 10)
    # I = np.array(I) / 255
    feature = encode(rgb_uv_hist(I))
    if outNum < len(wb_photo_finishing):
        wb_pf = random.sample(wb_photo_finishing, outNum)
        inds = []
        for j in range(outNum):
            inds.append(wb_photo_finishing.index(wb_pf[j]))
    else:
        wb_pf = wb_photo_finishing
        inds = list(range(0, len(wb_pf)))
    synthWBimages = []
    # ----------
    D_sq = np.einsum('ij, ij ->i', self_features, self_features)[:, None] + \
        np.einsum('ij, ij ->i', feature, feature) - 2 * self_features.dot(feature.T)
    # get smallest K distances
    idH = D_sq.argpartition(K, axis=0)[:K]
    dH = np.sqrt(np.take_along_axis(D_sq, idH, axis=0))
    weightsH = np.exp(-(np.power(dH, 2)) / (2 * np.power(sigma, 2)))  # compute weights
    weightsH = weightsH / sum(weightsH)  # normalize blending weights
    for i in range(len(inds)):  # for each of the retried training examples,
        ind = inds[i]  # for each WB & PF style,
        # generate a mapping function
        mf = sum(np.reshape(numpy.matlib.repmat(weightsH, 1, 27), (K, 1, 9, 3)) * mappingFuncs[(idH - 1) * 10 + ind, :])
        mf = mf.reshape(9, 3, order="F")  # reshape it to be 9 * 3
        synthWBimages.append(changeWB(I, mf))  # apply it!
    return synthWBimages, wb_pf


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

# WB & photo finishing styles
wb_photo_finishing = ['_F_AS', '_F_CS', '_S_AS', '_S_CS', 
                      '_T_AS', '_T_CS', '_C_AS', '_C_CS',
                      '_D_AS', '_D_CS']


# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

image_dir = os.path.join(base_path, '00_sample_images')

# image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
# print(f'num of images: {len(image_path_list)}')


# ----------
# img_file = os.path.join(image_dir, 'living_room.jpg')
# img_file = os.path.join(image_dir, 'bathroom_with_light_001.jpg')
# img_file = os.path.join(image_dir, 'bathroom_with_light_002.jpg')
# img_file = os.path.join(image_dir, 'bedroom.jpg')
# img_file = os.path.join(image_dir, 'bird_yellow.jpg')
# img_file = os.path.join(image_dir, 'strong_light.jpg')
# img_file = os.path.join(image_dir, 'road_day.png')
# img_file = os.path.join(image_dir, 'horseshoe_bend.png')

# img_file = os.path.join(image_dir, 'sun_flare_light_glare/driving_sun_glare.jpg')
img_file = os.path.join(image_dir, 'sun_flare_light_glare/girl_with_glare_light.png')



img = cv2.imread(img_file)

height = img.shape[0]
width = img.shape[1]
width_resize = 640

scale = width_resize / width

height_resize = int(height * scale)
width_resize = int(width * scale)

img = cv2.resize(img, (width_resize, height_resize))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------------------
# generate new images with different WB settings
# --------------------------------------------------------------------------------------------------

# should be <= 10
outNum = 9

outImgs, wb_pf = generateWbsRGB(img/255, outNum)


# ----------
fig = plt.figure(1)

grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )

for (i, image) in enumerate(outImgs):
    grid[i].imshow(np.asarray(image))
    grid[i].axis('off')
    grid[i].set_xticks([])
    grid[i].set_yticks([])


plt.tight_layout()
plt.show()

