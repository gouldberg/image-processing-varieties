
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import PIL.Image

from skimage import data, img_as_float
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error

from skimage.color import deltaE_ciede2000
from skimage import color

import albumentations as A

base_path = '/home/kswada/kw/image_processing'


####################################################################################################
# --------------------------------------------------------------------------------------------------
# BASICS:  skimage structural_similarity (SSIM)
# https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
# --------------------------------------------------------------------------------------------------

obj_img = data.camera()

print(obj_img)

# its gray (512, 512)
print(obj_img.shape)


# ----------
# convert to float (scale to range 0 - 1 by dividing 255
img = img_as_float(obj_img)

print(img)

# (512, 512)
print(img.shape)

rows, cols = img.shape


# ----------
# add noise
noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
rng = np.random.default_rng()
noise[rng.random(size=noise.shape) > 0.5] *= -1

print(noise)

# add noise or only plus constant
img_noise = img + noise
img_const = img + abs(noise)


# ----------
# no difference
mse_none = mean_squared_error(img, img)
ssim_none = structural_similarity(img, img, data_range=img.max() - img.min())

# compare to img_noise
mse_noise = mean_squared_error(img, img_noise)
ssim_noise = structural_similarity(img, img_noise, data_range=img_noise.max() - img_noise.min())

# compare to img_const
mse_const = mean_squared_error(img, img_const)
ssim_const = structural_similarity(img, img_const, data_range=img_const.max() - img_const.min())


# ----------
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(f'MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}')
ax[0].set_title('Original image')

ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(f'MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}')
ax[1].set_title('Image with noise')

ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(f'MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}')
ax[2].set_title('Image plus constant')

plt.tight_layout()
plt.show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# SSIM and color difference:  try other image data
# --------------------------------------------------------------------------------------------------

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

img_lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
img_lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

img_lab_sk1 = color.rgb2lab(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
img_lab_sk2 = color.rgb2lab(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

img_lab_f1 = cv2.cvtColor(img1.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
img_lab_f2 = cv2.cvtColor(img2.astype(np.float32)/255, cv2.COLOR_BGR2LAB)

# almost same with lab_f1 and lab_sk1 (np.float32 / 255)
print(img_lab1[0,10,0])
print(img_lab_f1[0,10,0])
print(img_lab_sk1[0,10,0])

img_f1 = img1.astype(np.float32) / 255
img_f2 = img2.astype(np.float32) / 255


# ----------
ssim_val, ssim_img = structural_similarity(img_gray1, img_gray2,
                                           data_range=255, full=True)

ssim_val2, ssim_img2 = structural_similarity(img1, img2,
                                             win_size=7, data_range=255,
                                             channel_axis=-1, full=True)

ssim_val3, ssim_img3 = structural_similarity(img_f1, img_f2,
                                             win_size=7, data_range=1.,
                                             channel_axis=-1, full=True)

ssim_val4, ssim_img4 = structural_similarity(img_f1, img_f2,
                                             channel_axis=-1, data_range=1., full=True,
                                             gaussian_weights=True, use_sample_covariance=True)

ssim_val5, ssim_img5 = structural_similarity(img_lab1, img_lab2,
                                             channel_axis=-1, full=True,
                                             gaussian_weights=True, use_sample_covariance=True)

ssim_val6, ssim_img6 = structural_similarity(img_lab_f1, img_lab_f2,
                                             channel_axis=-1, data_range=1., full=True,
                                             gaussian_weights=True, use_sample_covariance=True)


# KL = 1
KL = 100
color_diff = np.average(deltaE_ciede2000(img_lab1, img_lab2, kL=KL, kC=1, kH=1, channel_axis=-1))
color_diff2 = np.average(deltaE_ciede2000(img_lab_f1, img_lab_f2, kL=KL, kC=1, kH=1, channel_axis=-1))


# ----------
ssim_img_recov2 = np.clip(ssim_img2 * 255, 0, 255).astype('uint8')
ssim_img_recov4 = np.clip(ssim_img4 * 255, 0, 255).astype('uint8')
img_to_show = np.hstack([img1, img2, ssim_img_recov2, ssim_img_recov4])
PIL.Image.fromarray(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)).show()


print(f'SSIM value (gray):                  {ssim_val: .4f}')
print(f'SSIM value (multi channel):         {ssim_val2: .4f}')
print(f'SSIM value (multi channel, float):  {ssim_val3: .4f}')
print(f'SSIM value (multi channel2, float): {ssim_val4: .4f}')
print(f'SSIM value (LAB):                   {ssim_val4: .4f}')
print(f'SSIM value (LAB, float):            {ssim_val4: .4f}')
print(f'DeltaE CIEDE2000:                   {color_diff: .4f}')
print(f'DeltaE CIEDE2000 (float):           {color_diff2: .4f}')


# -->
# LAB base SSIM is not good ...
# In most case with strong data aug such as crop and rotate, LAB base results in high SSIM values.
# But on the other hands, simple SSIM results in very low values in case of simple rotate of gravel image ...


# DeltaE  CIEDE2000:
# Computing from image with np.float32 / 255 is better, and corresponds to skimage.color rgb2lab
# Also increasing scaling factor KL=1 to KL=100 to down-weight luminance impact
#  - Data Aug HueSaturation / InvertImg / ChannelShuffle:  larger difference
#  - Data Aug RandomBrightnessContrast:  smaller difference
#  - Data Aug (only) Rotate:  smaller difference
