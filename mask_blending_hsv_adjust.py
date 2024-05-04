
import os
import glob

import cv2
import numpy as np

import PIL.Image, PIL.ImageDraw, PIL.ImageFilter

import math
import matplotlib.pyplot as plt


base_path = '/home/kswada/kw/image_processing'


# --------------------------------------------------------------------------------------------------
# helper functions
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


def crop_image(img, ctr_x, ctr_y, crop_size):
    crop_size_x = crop_size[0]
    crop_size_y = crop_size[1]
    xmin = int(ctr_x - crop_size_x/2)
    xmax = int(ctr_x + crop_size_x/2)
    ymin = int(ctr_y - crop_size_y/2)
    ymax = int(ctr_y + crop_size_y/2)
    return img[ymin : ymax, xmin : xmax]


####################################################################################################
# --------------------------------------------------------------------------------------------------
# 2 images alpha blending and mask
# --------------------------------------------------------------------------------------------------

image_dir = os.path.join(base_path, '00_sample_images/prize/brassband_trading_badge')


# ----------
# load images
img1_path = os.path.join(image_dir, 'brassband_trading_badge_06.jpg')
img2_path = os.path.join(image_dir, 'brassband_trading_badge_07.jpg')

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

print(f'img1: {img1.shape}   img2: {img2.shape}')

# PIL.Image.fromarray(cv2.cvtColor(img1.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
# check

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# PIL.Image.fromarray(gray).show()

ret, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
# PIL.Image.fromarray(bin_img).show()

contours, hierarchy = cv2.findContours(
    bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)


# ----------
# bbox and contours
x1, y1, w1, h1, cont_max1, conts1, gray1, bin_img1 = get_bbox_by_inv_maxarea(img=img1, threshold=(250, 255))
x2, y2, w2, h2, cont_max2, conts2, gray2, bin_img2 = get_bbox_by_inv_maxarea(img=img2, threshold=(250, 255))


# ----------
# crop image by same size from center
ctr_x1 = int(x1 + w1/2)
ctr_y1 = int(y1 + h1/2)
ctr_x2 = int(x2 + w2/2)
ctr_y2 = int(y2 + h2/2)
# crop_size = (max(w1, w2) + 2, max(h1, h2) + 2)
crop_size = (max(w1, w2) + 1, max(h1, h2) + 1)

img1_cr = crop_image(img=img1, ctr_x=ctr_x1, ctr_y=ctr_y1, crop_size=crop_size)
img2_cr = crop_image(img=img2, ctr_x=ctr_x2, ctr_y=ctr_y2, crop_size=crop_size)


# ----------
# resize to fit for blending
img2_cr = cv2.resize(img2_cr, (img1_cr.shape[1], img1_cr.shape[0]))

print(f'img1: {img1.shape}   img2: {img2.shape}')
print(f'img1_crop: {img1_cr.shape}   img2_crop: {img2_cr.shape}')


# ----------
# again bbox and contours from cropped and center coordinated
x1, y1, w1, h1, cont_max1, conts1, gray1, bin_img1 = get_bbox_by_inv_maxarea(img=img1_cr, threshold=(250, 255))
x2, y2, w2, h2, cont_max2, conts2, gray2, bin_img2 = get_bbox_by_inv_maxarea(img=img2_cr, threshold=(250, 255))


# ----------
# generate mask
img1_m = np.zeros((img1_cr.shape[0], img1_cr.shape[1], 3))
img1_m = cv2.drawContours(img1_m, [cont_max1], 0, (255, 255, 255), -1)

img2_m = np.zeros((img2_cr.shape[0], img2_cr.shape[1], 3))
img2_m = cv2.drawContours(img2_m, [cont_max2], 0, (255, 255, 255), -1)

# if smoothing is required:
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# img1_m = cv2.morphologyEx(img1_mask, cv2.MORPH_OPEN, kernel, iterations=2)

img1_m = img1_m.astype('uint8')
img2_m = img2_m.astype('uint8')

# PIL.Image.fromarray(img1_m.astype('uint8')).show()
# PIL.Image.fromarray(img2_m.astype('uint8')).show()


# ----------
# alpha blend
alpha = 0.5
img_bl = cv2.addWeighted(img1_cr, alpha, img2_cr, 1 - alpha, 0.)

# PIL.Image.fromarray(cv2.cvtColor(img_bl.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
# masked blended image
img_bl_m = cv2.bitwise_and(img_bl, img1_m)

# PIL.Image.fromarray(cv2.cvtColor(img_blended_masked.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# HSV shift  (referencing to albumentations hsv_shift)
# --------------------------------------------------------------------------------------------------

def hsv_shift(img, shift_value):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    h_new, s_new, v_new = h, s, v
    h_shift, s_shift, v_shift = shift_value
    # ----------
    if h_shift != 0:
        h_new = cv2.add(h, h_shift)
        # h_new = np.mod(h_new, 360)  # OpenCV fails with negative values
    if s_shift != 0:
        # s_new = np.clip(cv2.add(s, s_shift), 0., 1.0)
        s_new = cv2.add(s, s_shift)
    if v_shift != 0:
        # v_new = np.clip(cv2.add(v, v_shift), 0., 1.0)
        v_new = cv2.add(v, v_shift)
    # ----------
    img = cv2.merge((h_new, s_new, v_new))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img, (h, s, v), (h_new, s_new, v_new)


# ----------
shift_value = (0, 0, -50)

img_hsv_shifted, (h, s, v), (h_new, s_new, v_new) = hsv_shift(img=img_bl_m, shift_value=shift_value)

print(f'hue       min: {np.min(h)}      max: {np.max(h)}      mean: {np.mean(h): .2f}      std: {np.std(h): .2f}')
print(f'hue_new   min: {np.min(h_new)}  max: {np.max(h_new)}  mean: {np.mean(h_new): .2f}  std: {np.std(h_new): .2f}')

print(f'sat       min: {np.min(s)}      max: {np.max(s)}      mean: {np.mean(s): .2f}      std: {np.std(s): .2f}')
print(f'sat_new   min: {np.min(s_new)}  max: {np.max(s_new)}  mean: {np.mean(s_new): .2f}  std: {np.std(s_new): .2f}')

print(f'val       min: {np.min(v)}      max: {np.max(v)}      mean: {np.mean(v): .2f}      std: {np.std(v): .2f}')
print(f'val_new   min: {np.min(v_new)}  max: {np.max(v_new)}  mean: {np.mean(v_new): .2f}  std: {np.std(v_new): .2f}')


# ----------
img_to_show = np.hstack([img_bl_m, img_hsv_shifted])

# PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# add shadow
# --------------------------------------------------------------------------------------------------

def add_shadow(img, coef=0.5):
    image_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = np.full(img.shape, 255).astype('uint8')
    # if red channel is hot, image's "Lightness" channel's brightness is lowered
    red_max_value_ind = mask[:, :, 0] == 255
    new_value = np.clip(image_hls[:, :, 1][red_max_value_ind] * coef, 0, 255)
    image_hls[:, :, 1][red_max_value_ind] = new_value
    # ----------
    image_bgr = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)
    return image_bgr


mode = 'l'

img_shadow = add_shadow(img=img_bl_m, coef=0.8)

img_to_show = np.hstack([img_bl_m, img_shadow])
PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# --------------------------------------------------------------------------------------------------
# histogram adjust to reference image distribution
# --------------------------------------------------------------------------------------------------

def get_hsvhls_stats_within_mask(img, mask, mode=('hsv', 'v')):
    within_mask = np.where(mask == 0, np.nan, 1)
    ret_val = np.nan
    if mode[0] == 'hsv':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        ch, val = 2, v
        if mode[1] == 'h':
            ch, val = 0, h
        elif mode[1] == 's':
            ch, val = 1, s
        else:
            pass
        ret_val = val * within_mask[..., ch]
    else:
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(img_hls)
        ch, val = 2, s
        if mode[1] == 'h':
            ch, val = 0, h
        elif mode[1] == 'l':
            ch, val = 1, l
        else:
            pass
        ret_val = val * within_mask[..., ch]
    return ret_val, np.nanmean(ret_val), np.nanmedian(ret_val), np.nanstd(ret_val)


def adjust_hsvhls(img, mask, dst_stats, mode=('hsv', 'v')):
    dst_mean = dst_stats[0]
    dst_std = dst_stats[1]
    # ----------
    within_mask = np.where(mask == 0, np.nan, 1)
    ret_img = img.copy()
    if mode[0] == 'hsv':
        ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(ret_img)
        ch, val, clip_val = 2, v, 255
        if mode[1] == 'h':
            ax, val, clip_val = 0, h, 179
        elif mode[1] == 's':
            ax, val, clip_val = 1, s, 255
        else:
            pass
        val = val * within_mask[..., ch]
        ret_img[..., ch] = np.clip((val - np.nanmean(val)) / np.nanstd(val) * dst_std + dst_mean, 0, clip_val)
        ret_img = cv2.cvtColor(ret_img, cv2.COLOR_HSV2BGR)
    else:
        ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(ret_img)
        ch, val, clip_val = 2, s, 255
        if mode[1] == 'h':
            ch, val, clip_val = 0, h, 179
        elif mode[1] == 'l':
            ch, val, clip_val = 1, l, 255
        else:
            pass
        val = val * within_mask[..., ch]
        ret_img[..., ch] = np.clip((val - np.nanmean(val)) / np.nanstd(val) * dst_std + dst_mean, 0, clip_val)
        ret_img = cv2.cvtColor(ret_img, cv2.COLOR_HLS2BGR)
    return ret_img


image_dir_for_ref1 = os.path.join(base_path, '00_sample_images')
# image_dir_for_ref2 = os.path.join(base_path, '00_sample_images/white_balance/blue')
# image_dir_for_ref2 = os.path.join(base_path, '00_sample_images/white_balance/red')


# ----------
img_ref1 = 'road_dark.png'
img_ref2 = 'road_day.png'
# img_ref2 = 'grass.png'
# img_ref2 = 'beach.png'
# img_ref2 = 'road_dark.png'
# img_ref2 = 'sunny_day_road.jpg'
# img_ref2 = 'nature_in_sunset.jpg'
# img_ref2 = 'almost_sunset.jpg'

img_ref1 = cv2.imread(os.path.join(image_dir_for_ref1, img_ref1))
img_ref2 = cv2.imread(os.path.join(image_dir_for_ref1, img_ref2))
# img_ref2 = cv2.imread(os.path.join(image_dir_for_ref2, img_ref2))

img_ref1_m = np.full(img_ref1.shape, 255).astype('uint8')
img_ref2_m = np.full(img_ref2.shape, 255).astype('uint8')


# ----------
# img_ref1 = cv2.bitwise_and(img1_cr, img1_m)
# img_ref2 = cv2.bitwise_and(img2_cr, img2_m)
# img_ref1_m = img1_m.copy()
# img_ref2_m = img2_m.copy()
# ----------


# PIL.Image.fromarray(cv2.cvtColor(img_ref1.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


#######################
# now set mode
mode = ('hsv', 'v')
# mode = ('hls', 'l')
#######################


# ----------
# get v stats of dst and src
dst_val1, dst_val1_mean, dst_val1_med, dst_val1_std = get_hsvhls_stats_within_mask(img=img_ref1, mask=img_ref1_m, mode=mode)
dst_val2, dst_val2_mean, dst_val2_med, dst_val2_std = get_hsvhls_stats_within_mask(img=img_ref2, mask=img_ref2_m, mode=mode)

val1, val1_mean, val1_med, val1_std = get_hsvhls_stats_within_mask(img=img1_cr, mask=img1_m, mode=mode)
val2, val2_mean, val2_med, val2_std = get_hsvhls_stats_within_mask(img=img2_cr, mask=img2_m, mode=mode)
valb, valb_mean, valb_med, valb_std = get_hsvhls_stats_within_mask(img=img_bl_m, mask=img1_m, mode=mode)

print(f'dst1  mean: {dst_val1_mean: .2f}  median: {dst_val1_med: .2f}  std: {dst_val1_std: .2f}')
print(f'dst2  mean: {dst_val2_mean: .2f}  median: {dst_val2_med: .2f}  std: {dst_val2_std: .2f}')

print(f'val1  mean: {val1_mean: .2f}  median: {val1_med: .2f}  std: {val1_std: .2f}')
print(f'val2  mean: {val2_mean: .2f}  median: {val2_med: .2f}  std: {val2_std: .2f}')
print(f'valb  mean: {valb_mean: .2f}  median: {valb_med: .2f}  std: {valb_std: .2f}')


# ----------
# v adjust
img1_cr_a1 = adjust_hsvhls(img=cv2.bitwise_and(img1_cr, img1_m), mask=img1_m, dst_stats=(dst_val1_mean, dst_val1_std), mode=mode)
img1_cr_a2 = adjust_hsvhls(img=cv2.bitwise_and(img1_cr, img1_m), mask=img1_m, dst_stats=(dst_val2_mean, dst_val2_std), mode=mode)

img2_cr_a1 = adjust_hsvhls(img=cv2.bitwise_and(img2_cr, img2_m), mask=img2_m, dst_stats=(dst_val1_mean, dst_val1_std), mode=mode)
img2_cr_a2 = adjust_hsvhls(img=cv2.bitwise_and(img2_cr, img2_m), mask=img2_m, dst_stats=(dst_val2_mean, dst_val2_std), mode=mode)

img_bl_a1 = adjust_hsvhls(img=img_bl_m, mask=img1_m, dst_stats=(dst_val1_mean, dst_val1_std), mode=mode)
img_bl_a2 = adjust_hsvhls(img=img_bl_m, mask=img1_m, dst_stats=(dst_val2_mean, dst_val2_std), mode=mode)


# ----------
# get stats after adjust
val1_a1, val1_mean_a1, val1_med_a1, val1_std_a1 = get_hsvhls_stats_within_mask(img=img1_cr_a1, mask=img1_m, mode=mode)
val1_a2, val1_mean_a2, val1_med_a2, val1_std_a2 = get_hsvhls_stats_within_mask(img=img1_cr_a2, mask=img1_m, mode=mode)

val2_a1, val2_mean_a1, val2_med_a1, val2_std_a1 = get_hsvhls_stats_within_mask(img=img2_cr_a1, mask=img2_m, mode=mode)
val2_a2, val2_mean_a2, val2_med_a2, val2_std_a2 = get_hsvhls_stats_within_mask(img=img2_cr_a2, mask=img2_m, mode=mode)

valb_a1, valb_mean_a1, valb_med_a1, valb_std_a1 = get_hsvhls_stats_within_mask(img=img_bl_a1, mask=img1_m, mode=mode)
valb_a2, valb_mean_a2, valb_med_a2, valb_std_a2 = get_hsvhls_stats_within_mask(img=img_bl_a2, mask=img1_m, mode=mode)


# ----------
img_to_show1 = np.hstack([cv2.bitwise_and(img1_cr, img1_m), img1_cr_a1, img1_cr_a2])
img_to_show2 = np.hstack([cv2.bitwise_and(img2_cr, img2_m), img2_cr_a1, img2_cr_a2])
img_to_show_bl = np.hstack([img_bl_m, img_bl_a1, img_bl_a2])

img_to_show = np.vstack([img_to_show1, img_to_show2, img_to_show_bl])
PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
# img_to_show_ref = np.hstack([img_ref1, img_ref2])
# PIL.Image.fromarray(cv2.cvtColor(img_to_show_ref.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


# ----------
# check stats
save_dir = os.path.join(base_path, '01_tmp')

plt.hist(dst_val1)
plt.show()
plt.savefig(os.path.join(save_dir, '01_01_dst_val1.png'))
plt.close()

plt.hist(val1)
plt.show()
plt.savefig(os.path.join(save_dir, '01_02_val1.png'))
plt.close()

plt.hist(val1_a1)
plt.show()
plt.savefig(os.path.join(save_dir, '01_03_val1_a1.png'))
plt.close()

plt.hist(val2)
plt.show()
plt.savefig(os.path.join(save_dir, '01_04_val2.png'))
plt.close()

plt.hist(val2_a1)
plt.show()
plt.savefig(os.path.join(save_dir, '01_05_val2_a1.png'))
plt.close()

plt.hist(valb)
plt.show()
plt.savefig(os.path.join(save_dir, '01_06_valb.png'))
plt.close()

plt.hist(valb_a1)
plt.show()
plt.savefig(os.path.join(save_dir, '01_07_valb_a1.png'))
plt.close()


# ----------
plt.hist(dst_val2)
plt.show()
plt.savefig(os.path.join(save_dir, '02_01_dst_val2.png'))
plt.close()

plt.hist(val1)
plt.show()
plt.savefig(os.path.join(save_dir, '02_02_val1.png'))
plt.close()

plt.hist(val1_a2)
plt.show()
plt.savefig(os.path.join(save_dir, '02_03_val1_a2.png'))
plt.close()

plt.hist(val2)
plt.show()
plt.savefig(os.path.join(save_dir, '02_04_val2.png'))
plt.close()

plt.hist(val2_a2)
plt.show()
plt.savefig(os.path.join(save_dir, '02_05_val2_a2.png'))
plt.close()

plt.hist(valb)
plt.show()
plt.savefig(os.path.join(save_dir, '02_06_valb.png'))
plt.close()

plt.hist(valb_a2)
plt.show()
plt.savefig(os.path.join(save_dir, '02_07_valb_a2.png'))
plt.close()
