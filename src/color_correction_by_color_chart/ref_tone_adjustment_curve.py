
import os
import glob
import shutil

import numpy as np
import cv2
import PIL.Image

import matplotlib.pyplot as plt

import pandas as pd

pd.set_option('display.max_rows', 80)
pd.set_option('display.max_columns', 20)


# --------------------------------------------------------------------------------------------------
# bezier curve
# --------------------------------------------------------------------------------------------------

t = np.linspace(0.0, 1.0, 256)

# this is coefficient for peak value level at lower t and higher t area
# if larger than 1.0, overshoot
# if smaller than 0.0, undershoot

# if both of low and high are small, almost close to t ** 3
# if low=0.42 and high=0.63, almost close to straight line
# if both are larger than 0.5, curve is exponentially decay
param_l = 0.7
param_h = 0.3

curve_bez = 3 * (1 - t) ** 2 * t * param_l + 3 * (1 - t) * t ** 2 * param_h + t ** 3
curve_bez0 = 3 * (1 - t) ** 2 * t * param_l
curve_bez1 = 3 * (1 - t) * t ** 2 * param_h
curve_bez2 = t ** 3


# ----------
plt.plot(t, curve_bez, label='all')
plt.plot(t, curve_bez0, label='curve 0')
plt.plot(t, curve_bez1, label='curve 1')
plt.plot(t, curve_bez2, label='curve 2')
plt.legend()
plt.show()

plt.close()


# ----------
# CURVE 0
# ----------
# this is param_l
# 1: peak level is around 0.43
coef_peak_val_list = np.linspace(0.0, 1.0, 11)

# 1: peak at t = 0.5
# default 2:  peak at around t = 0.35
coef_peak_shift_left_list = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]

for coef_peak_val in coef_peak_val_list:
# for coef_peak_shift_left in coef_peak_shift_left_list:
    # ----------
    curve_bez0 = 3 * (1 - t) ** 2 * t * coef_peak_val
    plt.plot(t, curve_bez0, label=f'curve0 {coef_peak_val: .2f}')
    # ----------
    # curve_bez0 = 3 * (1 - t) ** coef_peak_shift_left * t * 0.5
    # plt.plot(t, curve_bez0, label=f'curve0 {coef_peak_shift_left: .2f}')

plt.legend()
plt.show()

plt.close()


# ----------
# CURVE 1
# ----------
# this is param_h
# 1: peak level is around 0.43
coef_peak_val_list = np.linspace(0.0, 1.0, 11)

# 1: peak at t = 0.5
# default 2:  peak at around t = 0.65
coef_peak_shift_right_list = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]

# for coef_peak_val in coef_peak_val_list:
for coef_peak_shift_right in coef_peak_shift_right_list:
    # ----------
    # curve_bez1 = 3 * (1 - t) * t ** 2 * coef_peak_val
    # plt.plot(t, curve_bez1, label=f'curve1 {coef_peak_val: .2f}')
    # ----------
    curve_bez1 = 3 * (1 - t) * t ** coef_peak_shift_right * 0.5
    plt.plot(t, curve_bez1, label=f'curve1 {coef_peak_shift_right: .2f}')

plt.legend()
plt.show()

plt.close()



t = np.linspace(0.0, 1.0, 256)

def evaluate_bez(t):
    return 3 * (1 - t) ** 2 * t * par_l_r + 3 * (1 - t) * t ** 2 * par_h_r + t ** 3

par_l = 0.4
par_h = 0.6
bez = 3 * (1 - t) ** 2 * t * par_l + 3 * (1 - t) * t ** 2 * par_h + t ** 3

plt.plot(t, curve_)
plt.show()


plt.plot(x, y, label = "line 1")
plt.plot(y, x, label = "line 2")
plt.plot(x, np.sin(x), label = "curve 1")
plt.plot(x, np.cos(x), label = "curve 2")
plt.legend()
plt.show()

t = np.linspace(0.0, 1.0, 256)
save_fpath_tmp = os.path.join(save_folder, 'tmp')
par_l_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
par_h_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for par_l in par_l_list:
    for par_h in par_h_list:
        if par_l < par_h:
            print(f'{par_l} - {par_h}')
            fpath = os.path.join(save_fpath_tmp, f'curve_{par_l}_{par_h}.png')
            bez = 3 * (1 - t) ** 2 * t * par_l + 3 * (1 - t) * t ** 2 * par_h + t ** 3
            # bez = 4 * (1 - t) ** 2 * t * par_l + 4 * (1 - t) * t ** 2 * par_h + t ** 4
            # bez = 2 * (1 - t) ** 2 * t * par_l + 2 * (1 - t) * t ** 2 * par_h + t ** 2
            # bez = 5 * (1 - t) ** 2 * t * par_l + 5 * (1 - t) * t ** 2 * par_h + t ** 5
            plt.plot(t, bez)
            plt.savefig(fpath)
            plt.close()

#
# par_l_r, par_h_r = 0.4, 0.6
# par_l_g, par_h_g = 0.1, 0.9
# par_l_b, par_h_b = 0.2, 0.8
#
# evaluate_bez_r = np.vectorize(evaluate_bez_r)
# evaluate_bez_g = np.vectorize(evaluate_bez_g)
# evaluate_bez_b = np.vectorize(evaluate_bez_b)
#
# remapping_r = np.rint(evaluate_bez_r(t) * 255).astype(np.uint8)
# remapping_g = np.rint(evaluate_bez_g(t) * 255).astype(np.uint8)
# remapping_b = np.rint(evaluate_bez_b(t) * 255).astype(np.uint8)
#
# r_remap = cv2.LUT(r, lut=remapping_r)
# g_remap = cv2.LUT(g, lut=remapping_g)
# b_remap = cv2.LUT(b, lut=remapping_b)
# img_proc = cv2.merge([r_remap, g_remap, b_remap])
# PIL.Image.fromarray(img_proc.astype('uint8')).show()
# ##############################
