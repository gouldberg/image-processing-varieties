
import os
import cv2

import glob
import numpy as np

import shutil
import time

import PIL.Image, PIL.ImageDraw, PIL.ImageFilter


# --------------------------------------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------------------------------------

def add_alpha_channel(img):
    img = img.astype('uint8')
    alpha = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)
    return cv2.merge((b, g, r, alpha))


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


def crop_image(img, ctr_x, ctr_y, crop_size):
    crop_size_x = crop_size[0]
    crop_size_y = crop_size[1]
    xmin = int(ctr_x - crop_size_x/2)
    xmax = int(ctr_x + crop_size_x/2)
    ymin = int(ctr_y - crop_size_y/2)
    ymax = int(ctr_y + crop_size_y/2)
    return img[ymin : ymax, xmin : xmax]


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


####################################################################################################
# --------------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing/01_badge_illust'
# base_path = '/home/kswada/kw/image_processing/01_badge_illust_02'

base_img_dir = base_path

# card_img_path = os.path.join(base_path, 'kankore_front_kan/original/IMG_3015.JPG')
# card_img_mask_path = os.path.join(base_path, 'kankore_front_kan/mask/IMG_3015.png')

# card_img_orig_path = os.path.join(base_path, 'kankore_front_kan/original/sunny_day_building.jpg')
card_img_orig_path = os.path.join(base_path, 'kankore_front_kan/original/girl_with_glare_light.png')
card_img_path = os.path.join(base_path, 'kankore_front_kan/original/tmp.jpg')
card_img_mask_path = os.path.join(base_path, 'kankore_front_kan/mask/tmp.png')

prize_name_list = ['chara01', 'chara02', 'chara03', 'chara04', 'chara05', 'chara06', 'chara07', 'chara08', 'chara09']
# prize_name_list = ['chara01', 'chara02', 'chara03', 'chara04']


####################################################################################################
# --------------------------------------------------------------------------------------------------
# generate card mask and save
# --------------------------------------------------------------------------------------------------

# load card
card = cv2.imread(card_img_path)

print(f'card: {card.shape}')

# x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_inv_maxarea(img=card, threshold=(128, 255))
x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_maxarea(img=card, threshold=(128, 255))

# PIL.Image.fromarray(bin_img).show()


# ----------
# crop image by same size from center
# ctr_x = int(x + w/2)
# ctr_y = int(y + h/2)
# ctr_x = int(x + w/2)
# ctr_y = int(y + h/2)
# crop_size = (w, h)
#
# card_cr = crop_image(img=card, ctr_x=ctr_x, ctr_y=ctr_y, crop_size=crop_size)
# print(f'{crop_size} - {card_cr.shape}')

card_mk = np.zeros((card.shape[0], card.shape[1], 3))
card_mk = cv2.drawContours(card_mk, [cont_max], 0, (255, 255, 255), -1)


# card_mk = cv2.drawContours(card_cr, [cont_max], 0, (255, 255, 255), -1)
# ch_a = cv2.cvtColor(card_mk, cv2.COLOR_BGR2GRAY)
# b, g, r = cv2.split(card_mk)
# card_mk = cv2.merge((b, g, r, ch_a))

card_mk = card_mk.astype('uint8')

cv2.imwrite(card_img_mask_path, card_mk)

# PIL.Image.fromarray(cv2.cvtColor(card_cr.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
# PIL.Image.fromarray(card_mk).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# generate front card image and mask, and save  -->  card load
# --------------------------------------------------------------------------------------------------

# load original image for card
tmp = cv2.imread(card_img_orig_path)

print(f'tmp: {tmp.shape}')

ctr_x, ctr_y = int(tmp.shape[1]/2), int(tmp.shape[0]/2)
w, h = 300, 400

crop_xmin = int(ctr_x - w/2)
crop_xmax = int(ctr_x + w/2)
crop_ymin = int(ctr_y - h/2)
crop_ymax = int(ctr_y + h/2)

tmp_crop = tmp[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
tmp_crop_mask = np.full(tmp_crop.shape, 255).astype('uint8')

cv2.imwrite(card_img_path, tmp_crop)
cv2.imwrite(card_img_mask_path, tmp_crop_mask)


# --------------------------------------------------------------------------------------------------
# load card
# --------------------------------------------------------------------------------------------------

card = cv2.imread(card_img_path)
card_mk = cv2.imread(card_img_mask_path)

card_m = cv2.bitwise_and(card, card_mk)

# PIL.Image.fromarray(cv2.cvtColor(card_m.astype('uint8'), cv2.COLOR_BGR2RGB)).show()



####################################################################################################
# --------------------------------------------------------------------------------------------------
# alpha blending --> value adjustment --> save image and mask
# --------------------------------------------------------------------------------------------------

IS_ADJUST_VALUE = True
# IS_ADJUST_VALUE = False

if IS_ADJUST_VALUE:
    gen_img_dir = os.path.join(base_path, '02_gen_prize_adjust_y')
else:
    gen_img_dir = os.path.join(base_path, '02_gen_prize_adjust_n')

alpha_list = [0.2, 0.4, 0.6, 0.8]
gamma = 0

IS_ALPHA_GRADATION = True

adjust_mode = ('hsv', 'v')
# adjust_mode = ('hsl', 's')
# adjust_mode = ('hsl', 's')


# ----------
# calculate stats for card
card_val, card_val_mean, card_val_med, card_val_std = get_hsvhls_stats_within_mask(img=card, mask=card_m, mode=adjust_mode)

print(f'card  mean: {card_val_mean: .2f}  median: {card_val_med: .2f}  std: {card_val_std: .2f}')


# ----------
if os.path.exists(gen_img_dir):
    shutil.rmtree(gen_img_dir)

os.makedirs(gen_img_dir)
time.sleep(3)

for prize in prize_name_list:
    os.makedirs(os.path.join(gen_img_dir, prize + '_kan', 'original'))
    os.makedirs(os.path.join(gen_img_dir, prize + '_kan', 'mask'))
# ----------

img_idx = 9000

for idx in range(len(prize_name_list)):
    # -----------
    prize_name = prize_name_list[idx]
    img1_path = os.path.join(base_path, prize_name + '_kan', 'original', prize_name + '_1.JPG')
    img2_path = os.path.join(base_path, prize_name + '_kan', 'original', prize_name + '_2.JPG')
    save_orig_dir = os.path.join(gen_img_dir, prize_name + '_kan', 'original')
    save_mask_dir = os.path.join(gen_img_dir, prize_name + '_kan', 'mask')
    # -----------
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    print(f'img1: {img1.shape}   img2: {img2.shape}')
    # -----------
    # bbox and contours
    x1, y1, w1, h1, cont_max1, conts1, gray1, bin_img1 = get_bbox_by_inv_maxarea(img=img1, threshold=(250, 255))
    x2, y2, w2, h2, cont_max2, conts2, gray2, bin_img2 = get_bbox_by_inv_maxarea(img=img2, threshold=(250, 255))
    # -----------
    # crop image by same size from center
    ctr_x1 = int(x1 + w1/2)
    ctr_y1 = int(y1 + h1/2)
    ctr_x2 = int(x2 + w2/2)
    ctr_y2 = int(y2 + h2/2)
    crop_size = (max(w1, w2) + 2, max(h1, h2) + 2)
    img1_cr = crop_image(img=img1, ctr_x=ctr_x1, ctr_y=ctr_y1, crop_size=crop_size)
    img2_cr = crop_image(img=img2, ctr_x=ctr_x2, ctr_y=ctr_y2, crop_size=crop_size)
    # resize to fit for blending
    img2_cr = cv2.resize(img2_cr, (img1_cr.shape[1], img1_cr.shape[0]))
    # ----------
    # again bbox and contours from cropped and center coordinated
    x1, y1, w1, h1, cont_max1, conts1, gray1, bin_img1 = get_bbox_by_inv_maxarea(img=img1_cr, threshold=(250, 255))
    x2, y2, w2, h2, cont_max2, conts2, gray2, bin_img2 = get_bbox_by_inv_maxarea(img=img2_cr, threshold=(250, 255))
    # ----------
    # generate mask
    img1_mk = np.zeros((img1_cr.shape[0], img1_cr.shape[1], 3))
    img1_mk = cv2.drawContours(img1_mk, [cont_max1], 0, (255, 255, 255), -1)
    img2_mk = np.zeros((img2_cr.shape[0], img2_cr.shape[1], 3))
    img2_mk = cv2.drawContours(img2_mk, [cont_max2], 0, (255, 255, 255), -1)
    # if smoothing is required:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img1_mk = cv2.morphologyEx(img1_mk, cv2.MORPH_OPEN, kernel, iterations=2)
    img2_mk = cv2.morphologyEx(img2_mk, cv2.MORPH_OPEN, kernel, iterations=2)
    # ----------
    img1_mk = img1_mk.astype('uint8')
    img2_mk = img2_mk.astype('uint8')
    # ----------
    # image masked
    img1_m = cv2.bitwise_and(img1_cr, img1_mk)
    img2_m = cv2.bitwise_and(img2_cr, img2_mk)
    # ----------
    img1_m_a = img1_m.copy()
    img2_m_a = img2_m.copy()
    if IS_ADJUST_VALUE:
        val1, val1_mean, val1_med, val1_std = get_hsvhls_stats_within_mask(img=img1_cr, mask=img1_m, mode=adjust_mode)
        val2, val2_mean, val2_med, val2_std = get_hsvhls_stats_within_mask(img=img2_cr, mask=img2_m, mode=adjust_mode)
        print(f'val1  mean: {val1_mean: .2f}  median: {val1_med: .2f}  std: {val1_std: .2f}')
        print(f'val2  mean: {val2_mean: .2f}  median: {val2_med: .2f}  std: {val2_std: .2f}')
        img1_a = adjust_hsvhls(img=img1_m, mask=img1_m, dst_stats=(card_val_mean, card_val_std), mode=adjust_mode)
        img2_a = adjust_hsvhls(img=img2_m, mask=img2_m, dst_stats=(card_val_mean, card_val_std), mode=adjust_mode)
        img1_m_a = cv2.bitwise_and(img1_a, img1_mk)
        img2_m_a = cv2.bitwise_and(img2_a, img2_mk)
    img_idx += 1
    cv2.imwrite(os.path.join(save_orig_dir, f'IMG_{str(img_idx).zfill(4)}.JPG'), img1_m_a)
    cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), img1_mk)
    # cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), add_alpha_channel(img1_mk))
    img_idx += 1
    cv2.imwrite(os.path.join(save_orig_dir, f'IMG_{str(img_idx).zfill(4)}.JPG'), img2_m_a)
    cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), img2_mk)
    # cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), add_alpha_channel(img2_mk))
    # ----------
    # alpha blend
    for alpha in alpha_list:
        img_bl = cv2.addWeighted(img1_m, alpha, img2_m, 1 - alpha, gamma)
        # ----------
        # masked blended image
        img_bl_m = cv2.bitwise_and(img_bl, img1_mk)
        # img_to_show = np.vstack([
        #     np.hstack([img1_m, img1_mk]),
        #     np.hstack([img2_m, img2_mk]),
        #     np.hstack([img_bl_m, img1_mk])
        # ])
        # PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
        # ----------
        img_bl_a = img_bl.copy()
        if IS_ADJUST_VALUE:
            # valb, valb_mean, valb_med, valb_std = get_hsvhls_stats_within_mask(img=img_bl_m, mask=img1_m, mode=adjust_mode)
            # print(f'val1  mean: {val1_mean: .2f}  median: {val1_med: .2f}  std: {val1_std: .2f}')
            # print(f'val2  mean: {val2_mean: .2f}  median: {val2_med: .2f}  std: {val2_std: .2f}')
            # ----------
            # adjust value
            img_bl_a = adjust_hsvhls(img=img_bl_m, mask=img1_m, dst_stats=(card_val_mean, card_val_std), mode=adjust_mode)
            img_bl_m_a = cv2.bitwise_and(img_bl_a, img1_mk)
            # ----------
            # img_to_show = np.vstack([
            #     np.hstack([img1_m, img1_m_a]),
            #     np.hstack([img2_m, img2_m_a]),
            #     np.hstack([img_bl_m, img_bl_m_a])
            # ])
            # PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()
        img_idx += 1
        cv2.imwrite(os.path.join(save_orig_dir, f'IMG_{str(img_idx).zfill(4)}.JPG'), img_bl_a)
        cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), img1_mk)
        # cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), add_alpha_channel(img1_mk))
    if IS_ALPHA_GRADATION:
        w = img1_cr.shape[1]
        h = img1_cr.shape[0]
        alpha_w = np.linspace(0, 1, w).reshape(1, -1, 1)
        alpha_h = np.linspace(0, 1, h).reshape(-1, 1, 1)
        img_blw1 = np.array(img1_m * alpha_w + img2_m * (1 - alpha_w)).astype('uint8')
        img_blw2 = np.array(img2_m * alpha_w + img1_m * (1 - alpha_w)).astype('uint8')
        img_blh1 = np.array(img1_m * alpha_h + img2_m * (1 - alpha_h)).astype('uint8')
        img_blh2 = np.array(img2_m * alpha_h + img1_m * (1 - alpha_h)).astype('uint8')
        # ----------
        # masked blended image
        img_blw1_m = cv2.bitwise_and(img_blw1, img1_mk)
        img_blw2_m = cv2.bitwise_and(img_blw2, img1_mk)
        img_blh1_m = cv2.bitwise_and(img_blh1, img1_mk)
        img_blh2_m = cv2.bitwise_and(img_blh2, img1_mk)
        # ----------
        img_blw1_a = img_blw1.copy()
        img_blw2_a = img_blw2.copy()
        img_blh1_a = img_blw1.copy()
        img_blh2_a = img_blh2.copy()
        if IS_ADJUST_VALUE:
            img_blw1_a = adjust_hsvhls(img=img_blw1_m, mask=img1_m, dst_stats=(card_val_mean, card_val_std), mode=adjust_mode)
            img_blw2_a = adjust_hsvhls(img=img_blw2_m, mask=img1_m, dst_stats=(card_val_mean, card_val_std), mode=adjust_mode)
            img_blh1_a = adjust_hsvhls(img=img_blh1_m, mask=img1_m, dst_stats=(card_val_mean, card_val_std), mode=adjust_mode)
            img_blh2_a = adjust_hsvhls(img=img_blh2_m, mask=img1_m, dst_stats=(card_val_mean, card_val_std), mode=adjust_mode)
            img_blw1_m_a = cv2.bitwise_and(img_blw1_a, img1_mk)
            img_blw2_m_a = cv2.bitwise_and(img_blw2_a, img1_mk)
            img_blh1_m_a = cv2.bitwise_and(img_blh1_a, img1_mk)
            img_blh2_m_a = cv2.bitwise_and(img_blh2_a, img1_mk)
        img_idx += 1
        cv2.imwrite(os.path.join(save_orig_dir, f'IMG_{str(img_idx).zfill(4)}.JPG'), img_blw1_a)
        cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), img1_mk)
        # cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), add_alpha_channel(img1_mk))
        img_idx += 1
        cv2.imwrite(os.path.join(save_orig_dir, f'IMG_{str(img_idx).zfill(4)}.JPG'), img_blw2_a)
        cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), img1_mk)
        # cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), add_alpha_channel(img1_mk))
        img_idx += 1
        cv2.imwrite(os.path.join(save_orig_dir, f'IMG_{str(img_idx).zfill(4)}.JPG'), img_blh1_a)
        cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), img1_mk)
        # cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), add_alpha_channel(img1_mk))
        img_idx += 1
        cv2.imwrite(os.path.join(save_orig_dir, f'IMG_{str(img_idx).zfill(4)}.JPG'), img_blh2_a)
        cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), img1_mk)
        # cv2.imwrite(os.path.join(save_mask_dir, f'IMG_{str(img_idx).zfill(4)}.png'), add_alpha_channel(img1_mk))


####################################################################################################
# --------------------------------------------------------------------------------------------------
# check for composite
# --------------------------------------------------------------------------------------------------

dir_for_check = os.path.join(base_path, '03_check_for_composite')

if os.path.exists(dir_for_check):
    shutil.rmtree(dir_for_check)

os.makedirs(dir_for_check)
time.sleep(3)

# ----------
# card
scale_card = 1.0

card_scaled = cv2.resize(card_m, (int(card_m.shape[1] * scale_card), int(card_m.shape[0] * scale_card)))
# PIL.Image.fromarray(cv2.cvtColor(card_scaled.astype('uint8'), cv2.COLOR_BGR2RGB)).show()

ctr_x_comp = int(card_scaled.shape[1] / 2)
ctr_y_comp = int(card_scaled.shape[0] / 2)


# ----------
for prize_name in prize_name_list:
    img_orig_path_list = sorted(glob.glob(os.path.join(gen_img_dir, prize_name + '_kan', 'original', '*.JPG')))
    img_mask_path_list = sorted(glob.glob(os.path.join(gen_img_dir, prize_name + '_kan', 'mask', '*.png')))
    # ----------
    for i in range(len(img_orig_path_list)):
        # ----------
        img = cv2.imread(img_orig_path_list[i])
        mk = cv2.imread(img_mask_path_list[i])
        img_m = cv2.bitwise_and(img, mk)
        w, h = img_m.shape[1], img_m.shape[0]
        # ----------
        comp_xmin = int(ctr_x_comp - w/2)
        comp_xmax = int(ctr_x_comp + w/2)
        comp_ymin = int(ctr_y_comp - h/2)
        comp_ymax = int(ctr_y_comp + h/2)
        # ----------
        dst = card_scaled.copy()
        src_mk = np.full(card_scaled.shape, 0).astype('uint8')
        src_mk[comp_ymin : comp_ymax, comp_xmin : comp_xmax] = img_m
        dst[:] = np.where(src_mk == 0, dst, src_mk)
        cv2.imwrite(os.path.join(dir_for_check, f'{prize_name}_{str(i).zfill(4)}.JPG'), dst)
        # ----------
        # img_to_show = np.hstack([card_scaled, dst])
        # PIL.Image.fromarray(cv2.cvtColor(img_to_show.astype('uint8'), cv2.COLOR_BGR2RGB)).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# check for composite:  compare various cases
# --------------------------------------------------------------------------------------------------

dir_for_check_list = [
    os.path.join(base_path, '03_check_for_composite_original_front'),
    os.path.join(base_path, '03_check_for_composite_sunnyday_building'),
    os.path.join(base_path, '03_check_for_composite_girlwithblarelight')
]

dir_for_compare = os.path.join(base_path, '04_compare_composite')

if os.path.exists(dir_for_compare):
    shutil.rmtree(dir_for_compare)

os.makedirs(dir_for_compare)
time.sleep(3)


# ----------
img_name_list = sorted(glob.glob(os.path.join(dir_for_check_list[0], '*.JPG')))

img_basename_list = []

for fpath in img_name_list:
    img_basename_list.append(os.path.basename(fpath))


# ----------
for i in range(len(img_basename_list)):
    img0 = cv2.imread(os.path.join(dir_for_check_list[0], img_basename_list[i]))
    img1 = cv2.imread(os.path.join(dir_for_check_list[1], img_basename_list[i]))
    img2 = cv2.imread(os.path.join(dir_for_check_list[2], img_basename_list[i]))
    # ----------
    img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]))
    img2 = cv2.resize(img2, (img0.shape[1], img0.shape[0]))
    # ----------
    img_output = np.hstack([img0, img1, img2])
    cv2.imwrite(os.path.join(dir_for_compare, img_basename_list[i]), img_output)





