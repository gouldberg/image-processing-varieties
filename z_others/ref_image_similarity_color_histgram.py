
import os
import numpy as np
import cv2

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import PIL.Image

from skimage import color

import albumentations as A

from scipy.spatial import distance as dist

base_path = '/home/kswada/kw/image_processing'


# --------------------------------------------------------------------------------------------------
# color histgram distance
# functions
# --------------------------------------------------------------------------------------------------

def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])


def calc_hist_dist(img1, img2, channels=[0,1,2], histSize=[8,8,8], ranges=[0, 256, 0, 256, 0,256]):
    # ----------
    hist1 = cv2.calcHist([img1], channels=channels, mask=None, histSize=histSize, ranges=ranges)
    hist2 = cv2.calcHist([img2], channels=channels, mask=None, histSize=histSize, ranges=ranges)
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    # ----------
    results = {}
    results['Intersection'] = np.round(cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT), 4)
    results['Chi-Sqaured'] = np.round(chi2_distance(hist1, hist2), 4)
    return results


# --------------------------------------------------------------------------------------------------
# set 2 image paths
# --------------------------------------------------------------------------------------------------

img_dir = '/home/kswada/kw/image_processing/00_sample_images/ssim'

# img_fname1 = 'test2-1.png'
# img_fname2 = 'test2-2.png'

# img_fname1 = 'test3-orig.jpg'
# img_fname2 = 'test3-cro.jpg'
# img_fname2 = 'test3-lig.jpg'
# img_fname2 = 'test3-rot.jpg'

# img_fname1 = 'camera.png'
# img_fname2 = 'camera_noise.png'  # SSIM is 0.15 (by skimage.metrics.structural_similarity)
# img_fname2 = 'camera_const.png'  # SSIM is 0.85 (by skimage.metrics.structural_similarity)


# ----------
# img_dir = '/home/kswada/kw/image_processing/00_sample_images/prize/brassband_trading_badge'
# img_fname1 = 'brassband_trading_badge_01.jpg'
# img_fname2 = 'brassband_trading_badge_02.jpg'


# ----------
# img_dir = '/home/kswada/kw/image_processing/00_sample_images/skimage_data'
# img_fname1 = 'chelsea.png'
# img_fname2 = 'chelsea.png'
# img_fname1 = 'chessboard_RGB.png'
# img_fname2 = 'chessboard_RGB.png'
# img_fname1 = 'chessboard_GRAY.png'
# img_fname2 = 'chessboard_GRAY.png'
# img_fname1 = 'ihc.png'
# img_fname2 = 'ihc.png'
# img_fname1 = 'motorcycle_left.png'
# img_fname2 = 'motorcycle_left.png'
# img_fname2 = 'motorcycle_right.png'
# img_fname1 = 'page.png'
# img_fname2 = 'page.png'
# img_fname1 = 'grass.png'
# img_fname2 = 'grass.png'
# img_fname1 = 'gravel.png'
# img_fname2 = 'gravel.png'
# img_fname1 = 'logo.png'
# img_fname2 = 'logo.png'


# ----------
# img_dir = '/home/kswada/kw/image_processing/00_sample_images/cushion/pacman'
# # # img_fname1 = 'cushion_galaxy_donuts_ghostbig_pacman_blue.jpg'
# img_fname1 = 'cushion_galaxy_donuts_ghostbig_pacman_orange.jpg'
# # img_fname1 = 'cushion_galaxy_donuts_ghostbig_pacman_pink.jpg'
# img_fname2 = 'cushion_pacman_pink.jpg'
# # img_fname2 = 'cushion_pacman_ura_blue.jpg'


# ----------
img_dir = '/home/kswada/kw/image_processing/00_sample_images/cushion'
img_fname1 = 'cushion_01.jpg'
img_fname2 = 'cushion_03.jpg'
# img_fname1 = 'cushion_blown.jpg'
# # # img_fname2 = 'cushion_07.jpg'
# img_fname2 = 'cushion_green_gray.jpg'


# ----------
img_path = os.path.join(img_dir, img_fname1)
img_path2 = os.path.join(img_dir, img_fname2)


# --------------------------------------------------------------------------------------------------
# compute SSIM and color difference
# --------------------------------------------------------------------------------------------------

img1 = cv2.imread(img_path)
img2 = cv2.imread(img_path2)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))


# ----------
transform = A.Compose([
    # A.HorizontalFlip(p=1.0),
    # A.RandomBrightnessContrast(brightness_limit=(-0.2*2, 0.2*2), contrast_limit=(-0.2*2, 0.2*2), p=1.0),
    # A.Rotate((45, 45), p=1.0),
    # A.HueSaturationValue((-20*4, 20*4), (-30*4, 30*4), (-20*4, 20*4), p=1.0),
    # A.ChannelShuffle(p=1.0),
    # A.InvertImg(p=1.0),
    # A.RandomSizedCrop((200, 200), height=img1.shape[0], width=img1.shape[1], p=1.0)
])


img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
transformed = transform(image=img_rgb)
trans_img = transformed["image"]
img2 = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
# PIL.Image.fromarray(trans_img).show()


img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img_lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
img_lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

hist_res_rgb = calc_hist_dist(img_rgb1, img_rgb2, channels=[0, 1, 2], histSize=[16, 16, 16], ranges=[0, 256, 0, 256, 0, 256])
hist_res_lab = calc_hist_dist(img_lab1, img_lab2, channels=[1, 2], histSize=[16, 16], ranges=[0, 256, 0, 256])
