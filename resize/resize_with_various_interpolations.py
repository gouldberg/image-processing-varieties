
sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import cv2
import numpy as np

import os



# --------------------------------------------------------------------------------------
# createPattern(), getResize()
# --------------------------------------------------------------------------------------

def createPattern():
    square = 200
    border = 2
    size = np.array([square, square, 3])
    color = np.array([255., 255., 255.])
    img = np.full(size, color, dtype=np.uint8)

    # 斜め格子パターンを作成
    borderColor = (255, 0, 0)
    for wh in range(0, square, 10):
        cv2.line(img=img, pt1=(wh, 0), pt2=(0, wh), color=borderColor, thickness=border, lineType=cv2.LINE_AA)
    for wh in range(0, square, 10):
        cv2.line(img=img, pt1=(square, wh), pt2=(wh, square), color=borderColor, thickness=border, lineType=cv2.LINE_AA)

    borderColor = (255, 0, 255)
    for wh in range(square, 0, -10):
        cv2.line(img=img, pt1=(wh, 0), pt2=(square, square - wh), color=borderColor, thickness=border, lineType=cv2.LINE_AA)

    return img



# --------------------------------------------------------------------------------------
# createPattern()
# --------------------------------------------------------------------------------------

img = createPattern()



# ----------
basePixSize = 600  # 縦横で大きい辺の変更したいサイズ

height = img.shape[0]

width = img.shape[1]

largeSize = max(height, width)  # 大きい方の辺のサイズ

resizeRate = basePixSize / largeSize  # 変更比率を計算



# ----------
# INTER_LINEAR (バイリニア補間)
# interpolationの指定がなければこの補間手法がデフォルトとして実行される

dst1 = cv2.resize(img.copy(), (int(width * resizeRate), int(height * resizeRate)),
                  interpolation=cv2.INTER_LINEAR)


# INTER_NEAREST (最近傍補間)
dst2 = cv2.resize(img.copy(), (int(width * resizeRate), int(height * resizeRate)),
                  interpolation=cv2.INTER_NEAREST)


# INTER_CUBIC (4×4 の近傍領域を利用するバイキュービック補間)
dst3 = cv2.resize(img.copy(), (int(width * resizeRate), int(height * resizeRate)),
                  interpolation=cv2.INTER_CUBIC)


# INTER_AREA (平均画素法)
dst4 = cv2.resize(img.copy(), (int(width * resizeRate), int(height * resizeRate)),
                 interpolation=cv2.INTER_AREA)


# INTER_LANCZOS4 (8×8 の近傍領域を利用する ランチョス法の補間)
# GPUを使用するとエラーになる
dst5 = cv2.resize(img.copy(), (int(width * resizeRate), int(height * resizeRate)),
                  interpolation=cv2.INTER_LANCZOS4)



# ----------
cv2.imshow('5  INTER_LANCZOS4', dst5)
cv2.imshow('4  INTER_AREA', dst4)
cv2.imshow('3  INTER_CUBIC', dst3)
cv2.imshow('2  INTER_NEAREST', dst2)
cv2.imshow('1  INTER_LINEAR', dst1)
cv2.imshow('0  Original', img)
cv2.waitKey(0)


# -->
# INTER_NEAREST is poor quality
# INTER_CUBIC and INTER_LANCZOS4 is best quality  (INTER_LANCZOS4 takes time)



# ----------
cv2.destroyAllWindows()

