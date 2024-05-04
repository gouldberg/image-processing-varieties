
sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import cv2
import numpy as np
import time

import os



# --------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------

img_path = "C:\\Users\\kouse\\Desktop\\image_data\\images"

img_file = os.path.join(img_path, "terrace_IMG_0379.jpg")


# ----------
img = cv2.imread(img_file)

import imutils

img = imutils.resize(img, width=600)


clone = img.copy()



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

