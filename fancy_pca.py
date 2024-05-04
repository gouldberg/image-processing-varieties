
import os
import glob

import numpy as np
import math
import random

import cv2
import PIL.Image


# ----------
# reference:
# https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py
# http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf


####################################################################################################
# --------------------------------------------------------------------------------------------------
# fancy pca
#   - perform PCA on the set of RGB pixel values
#   - add multiples of the found principal components, with magnitudes proportional to the corresponding eigenvalues
#     times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1
#   - This scheme approximately captures an important property of natural images, namely, that
#     object intensity is invariant to changes in the intensity and color of the illumination
# --------------------------------------------------------------------------------------------------

def fancy_pca(img, alpha=0.1):
    """Perform 'Fancy PCA' augmentation from:
    http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    Args:
        img (numpy.ndarray): numpy array with (h, w, rgb) shape, as ints between 0-255
        alpha (float): how much to perturb/scale the eigen vecs and vals
                the paper used std=0.1
    Returns:
        numpy.ndarray: numpy image-like array as uint8 range(0, 255)
    """
    # if not is_rgb_image(img) or img.dtype != np.uint8:
    #     raise TypeError("Image must be RGB image in uint8 format.")
    # ----------
    orig_img = img.astype(float).copy()
    img = img / 255.0  # rescale to 0 to 1 range
    # ----------
    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)
    # ----------
    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)
    # ----------
    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)
    # ----------
    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)
    # ----------
    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]
    # ----------
    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))
    # ----------
    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    # alpha = np.random.normal(0, alpha_std)
    # ----------
    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]
    # ----------
    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)
    # ----------
    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx] * 255
    # ----------
    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)
    # ----------
    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)
    # ----------
    return orig_img


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing/data_augmentation'

image_dir = os.path.join(base_path, 'sample_images')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')


# ----------
index = 0
img = cv2.imread(image_path_list[index])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------------------
# fancy pca
# --------------------------------------------------------------------------------------------------

alpha = 0.1

img_pca = fancy_pca(img=img, alpha=alpha)

# PIL.Image.fromarray(img).show()
PIL.Image.fromarray(img_pca).show()


