
sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import cv2
import numpy as np
import time

import os



# --------------------------------------------------------------------------------------
# create base Mat
# --------------------------------------------------------------------------------------

width = 600

height = 480

size = (height, width, 3)  # 縦480ピクセル 横640ピクセル 3チャンネル


# generate Mat
img = np.zeros(size, dtype=np.uint8)



# --------------------------------------------------------------------------------------
# create gradation
# --------------------------------------------------------------------------------------

rate = 255 / width

print(rate)

for h in range(0, height):
    for w in range(0, width):
        data = int(w * rate)
        img[h, w] = [data, data, data]



# --------------------------------------------------------------------------------------
# overlay coloured image
# --------------------------------------------------------------------------------------

clone = img.copy()


# ----------
red = np.array([0., 0., 255.])

green = np.array([0., 255., 0.])

blue = np.array([255., 0., 0.])

cyan = np.array([255., 255., 0.])

magenta = np.array([255., 0., 255])

yellow = np.array([0., 255., 255.])


# ----------
# pt1: startX, startY
# pt2: endX, endY
# thickness = -1:  inner area is painted

cv2.rectangle(img=clone, pt1=(0, 0), pt2=(200, 200), color=red, thickness=-1)
cv2.rectangle(img=clone, pt1=(200, 0), pt2=(400, 200), color=green, thickness=-1)
cv2.rectangle(img=clone, pt1=(400, 0), pt2=(600, 200), color=blue, thickness=-1)

cv2.rectangle(img=clone, pt1=(0, 200), pt2=(200, 400), color=cyan, thickness=-1)
cv2.rectangle(img=clone, pt1=(200, 200), pt2=(400, 400), color=magenta, thickness=-1)
cv2.rectangle(img=clone, pt1=(400, 200), pt2=(600, 400), color=yellow, thickness=-1)



# --------------------------------------------------------------------------------------
# Binarize
# --------------------------------------------------------------------------------------

# gray scaled
gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)


# ----------
# APPLY TO ORIGINAL IMAGE
# 閾値(thresh)以下の値は0に，それ以外はmaxValで指定した値になる

ret, c_dst = cv2.threshold(clone, thresh=128, maxval=255, type=cv2.THRESH_BINARY)

ret, c_dst_inv = cv2.threshold(clone, thresh=128, maxval=255, type=cv2.THRESH_BINARY_INV)

ret, c_dst_tozero = cv2.threshold(clone, thresh=128, maxval=255, type=cv2.THRESH_TOZERO)

ret, c_dst_tozeroinv = cv2.threshold(clone, thresh=128, maxval=255, type=cv2.THRESH_TOZERO_INV)

ret, c_dst_trunc = cv2.threshold(clone, thresh=128, maxval=255, type=cv2.THRESH_TRUNC)



# ----------
# APPLY TO GLAY SCALED IMAGE
ret, g_dst = cv2.threshold(gray, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
ret, g_dst_inv = cv2.threshold(gray, thresh=128, maxval=255, type=cv2.THRESH_BINARY_INV)

# 大津アルゴリズムでは thresh, maxvalは無視されてしきい値は自動で設定される
ret, g_dst_otsu = cv2.threshold(gray, thresh=0, maxval=255,
                                type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# トライアングルアルゴリズムでは threshは無視されてしきい値は自動で設定される
ret, g_dst_triangle = cv2.threshold(gray, thresh=0, maxval=255,
                         type=cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

ret, g_dst_tozero = cv2.threshold(gray, thresh=128, maxval=255, type=cv2.THRESH_TOZERO)

ret, g_dst_tozeroinv = cv2.threshold(gray, thresh=128, maxval=255, type=cv2.THRESH_TOZERO_INV)

ret, g_dst_trunc = cv2.threshold(gray, thresh=128, maxval=255, type=cv2.THRESH_TRUNC)



# ----------
cv2.imshow('1-7 Gray --> binary trunc', g_dst_trunc)
cv2.imshow('0-7 Orig --> binary trunc', c_dst_trunc)

cv2.imshow('1-6 Gray --> binary tozero inv', g_dst_tozeroinv)
cv2.imshow('0-6 Orig --> binary tozero inv', c_dst_tozeroinv)

cv2.imshow('1-5 Gray --> binary tozero', g_dst_tozero)
cv2.imshow('0-5 Orig --> binary tozero', c_dst_tozero)

cv2.imshow('1-4 Gray --> binary triangle', g_dst_triangle)

cv2.imshow('1-3 Gray --> binary otsu', g_dst_otsu)

cv2.imshow('1-2 Gray --> binary inv', g_dst_inv)
cv2.imshow('0-2 Orig --> binary inv', c_dst_inv)

cv2.imshow('1-1 Gray --> binary', g_dst)
cv2.imshow('0-1 Orig --> binary', c_dst)

cv2.imshow('1-0 Gray', gray)
cv2.imshow('0-0 Original', clone)
cv2.waitKey(0)


# ----------
cv2.destroyAllWindows()

