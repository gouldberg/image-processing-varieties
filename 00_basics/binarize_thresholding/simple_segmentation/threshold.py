# USAGE
# python threshold.py --image images/skateboard_decks.png --threshold 245

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import argparse
import cv2

import os



# ------------------------------------------------------------------------------------------
# set arguments
# ------------------------------------------------------------------------------------------

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
# 	help = "Path to the image to be thresholded")
# ap.add_argument("-t", "--threshold", type = int, default = 128,
# 	help = "Threshold value")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20140908_thresholding-simple-segmentation"



# ------------------------------------------------------------------------------------------
# load image
# ------------------------------------------------------------------------------------------

img_file = os.path.join(base_path, "images\\skateboard_decks.png")


image = cv2.imread(img_file)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# ------------------------------------------------------------------------------------------
# initialize the list of threshold methods
# ------------------------------------------------------------------------------------------

methods = [
	("THRESH_BINARY", cv2.THRESH_BINARY),
	("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
	("THRESH_TRUNC", cv2.THRESH_TRUNC),
	("THRESH_TOZERO", cv2.THRESH_TOZERO),
	("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]


# THRESH_TRUNC:
#  - leaves the pixel intensities as they are
#  if the source pixel is not greater than the supplied threshold.

# THRESH_TOZERO:
#  - sets the source pixel to zero
#  if the source pixel is not greater than the supplied threshold:


# ------------------------------------------------------------------------------------------
# Thresholding
# ------------------------------------------------------------------------------------------

thre = 128


for (threshName, threshMethod) in methods:
	# threshold the image and show it
	(T, thresh) = cv2.threshold(gray, thre, 255, threshMethod)
	cv2.imshow(threshName, thresh)
	cv2.waitKey(0)


