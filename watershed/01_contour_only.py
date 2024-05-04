# USAGE
# python contour_only.py --image images/coins_01.png

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

# from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import cv2

import os



# -------------------------------------------------------------------------------------
# set arguments
# -------------------------------------------------------------------------------------

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20151102_watershed"



# -------------------------------------------------------------------------------------
# load image
# -------------------------------------------------------------------------------------

img_file = os.path.join(base_path, "images\\coins_01.png")
# img_file = os.path.join(base_path, "images\\coins_02.png")
# img_file = os.path.join(base_path, "images\\coins_03.png")
# img_file = os.path.join(base_path, "images\\pills_01.png")
# img_file = os.path.join(base_path, "images\\pills_02.png")
# img_file = "C:\\Users\\kouse\\Desktop\\image_data\\images\\family\\family_HU_P1010575.jpg"

image = cv2.imread(img_file)

# image = imutils.resize(image, width=600)


# -------------------------------------------------------------------------------------
# pyramid mean shift filtering
# -------------------------------------------------------------------------------------

# perform pyramid mean shift filtering
# to aid the thresholding step

shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)



# -------------------------------------------------------------------------------------
# Threshodling
# -------------------------------------------------------------------------------------

# convert the mean shift image to grayscale, then apply
# Otsu's thresholding

gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]



# -------------------------------------------------------------------------------------
# find contours
# -------------------------------------------------------------------------------------

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)[-2]


print("[INFO] {} unique contours found".format(len(cnts)))



# -------------------------------------------------------------------------------------
# display contours
# -------------------------------------------------------------------------------------

# loop over the contours

for (i, c) in enumerate(cnts):
	# draw the contour
	((x, y), _) = cv2.minEnclosingCircle(c)
	cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)



# ----------
# show the output image
cv2.imshow("03 Image+Contours", image)
cv2.imshow("02 Thresholded", thresh)
cv2.imshow("01 Shifted", shifted)
cv2.waitKey(0)

