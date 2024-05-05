# USAGE
# python rotate_simple.py --image images/saratoga.jpg

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import numpy as np
import argparse
import imutils
import cv2

import os


# ------------------------------------------------------------------------------------------
# rotate(), rotate_bound()
# ------------------------------------------------------------------------------------------

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # !!! compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# ------------------------------------------------------------------------------------------
# set arguments
# ------------------------------------------------------------------------------------------

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the image file")
# args = vars(ap.parse_args())

base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20170102_opencv-rotate"



# ------------------------------------------------------------------------------------------
# load image
# ------------------------------------------------------------------------------------------

img_file = os.path.join(base_path, "images\\pill_01.png")
img_file = os.path.join(base_path, "images\\saratoga.jpg")


image = cv2.imread(img_file)

print(image.shape)



# ------------------------------------------------------------------------------------------
# rotation 1
# ------------------------------------------------------------------------------------------

# loop over the rotation angles
for angle in np.arange(0, 360, 15):
	rotated = rotate(image, angle)
	cv2.imshow("Rotated (Problematic)", rotated)
	cv2.waitKey(0)



# ------------------------------------------------------------------------------------------
# rotation 2
# ------------------------------------------------------------------------------------------

# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
for angle in np.arange(0, 360, 15):
	rotated = rotate_bound(image, angle)
	cv2.imshow("Rotated (Correct)", rotated)
	cv2.waitKey(0)




# The cv2.getRotationMatrix2D function doesn’t care if we would like the entire rotated image to kept.
# It doesn’t care if the image is cut off.
