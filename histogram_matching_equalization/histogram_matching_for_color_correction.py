
import os
import glob

import numpy as np
import math
import random

import cv2
import PIL.Image

# from scipy.spatial import distance as dist


####################################################################################################
# -----------------------------------------------------------------------------------------------
# find_color_card()
# -----------------------------------------------------------------------------------------------

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # ----------
    # grab the left-most and right-most points from the sorted x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # ----------
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # ----------
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    # ----------
    # D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    D = np.sum((tl - rightMost)**2, axis=1)**0.5
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # ----------
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # ----------
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # ----------
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # ----------
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # ----------
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # ----------
    # return the warped image
    return warped


def find_color_card(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image
    # ----------
    # arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    # arucoParams = cv2.aruco.DetectorParameters_create()
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    # ----------
    # try to extract the coordinates of the color correction card
    # otherwise, we've found the four ArUco markers, so we can
    # continue by flattening the ArUco IDs list
    ids = ids.flatten()
    # ----------
    # extract the top-left marker
    i = np.squeeze(np.where(ids == 923))
    topLeft = np.squeeze(corners[i])[0]
    # ----------
    # extract the top-right marker
    i = np.squeeze(np.where(ids == 1001))
    topRight = np.squeeze(corners[i])[1]
    # ----------
    # extract the bottom-right marker
    i = np.squeeze(np.where(ids == 241))
    bottomRight = np.squeeze(corners[i])[2]
    # ----------
    # extract the bottom-left marker
    i = np.squeeze(np.where(ids == 1007))
    bottomLeft = np.squeeze(corners[i])[3]
    # ----------
    # build our list of reference points and apply a perspective
    # transform to obtain a top-down, birds-eye-view of the color
    # matching card
    cardCoords = np.array([topLeft, topRight, bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoords)
    # ----------
    # return the color matching card to the calling function
    return card


# -----------------------------------------------------------------------------------------------
# histogram matching by numpy
# -----------------------------------------------------------------------------------------------

def histogram_matching(src, ref):
    src_lookup = src.reshape(-1)
    src_counts = np.bincount(src_lookup)
    tmpl_counts = np.bincount(ref.reshape(-1))
    # ----------
    # omit values where the count was 0
    tmpl_values = np.nonzero(tmpl_counts)[0]
    tmpl_counts = tmpl_counts[tmpl_values]
    # ----------
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / src.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / ref.size
    # ----------
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(src.shape)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir = os.path.join(base_path, '00_sample_images/color_correction')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')


# ----------
img = cv2.imread(os.path.join(image_dir, '01.jpg'))
# img = cv2.imread(os.path.join(image_dir, '02.jpg'))
# img = cv2.imread(os.path.join(image_dir, '03.jpg'))

ref = cv2.imread(os.path.join(image_dir, 'reference.jpg'))


# ----------
height = img.shape[0]
width = img.shape[1]
width_resize = 640

scale = width_resize / width

height_resize = int(height * scale)
width_resize = int(width * scale)

img = cv2.resize(img, (width_resize, height_resize))
ref = cv2.resize(ref, (width_resize, height_resize))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------------------------
# find the color matching card
# -----------------------------------------------------------------------------------------------

refCard = cv2.resize(find_color_card(ref), (width_resize, height_resize))

imageCard = cv2.resize(find_color_card(img), (width_resize, height_resize))


img_to_show = np.hstack([img, ref, imageCard, refCard])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()


# ----------
# apply histogram matching from the color matching card in the
# reference image to the color matching card in the input image

matched = np.zeros(img.shape)
matched[..., 0] = histogram_matching(imageCard[..., 0], refCard[..., 0])
matched[..., 1] = histogram_matching(imageCard[..., 1], refCard[..., 1])
matched[..., 2] = histogram_matching(imageCard[..., 2], refCard[..., 2])

img_to_show = np.hstack([img, ref, matched])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()
