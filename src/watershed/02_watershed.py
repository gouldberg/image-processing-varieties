# USAGE
# python watershed.py --image images/coins_01.png

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import cv2

import os



# -----------------------------------------------------------------------------------------
# set arguments
# -----------------------------------------------------------------------------------------

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
# find local peak of distance
# -------------------------------------------------------------------------------------

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map

D = ndimage.distance_transform_edt(thresh)

localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=thresh)



# ----------
print(np.min(D))
print(np.max(D))

D0 = ((np.max(D) - D) / np.max(D) * 255).astype("uint8")


# -------------------------------------------------------------------------------------
# connected component analysis
# -------------------------------------------------------------------------------------

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then apply the Watershed algorithm

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

labels = watershed(-D, markers, mask=thresh)

print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))



# -------------------------------------------------------------------------------------
# Watershed
# -------------------------------------------------------------------------------------

# loop over the unique labels returned by the Watershed
# algorithm

for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue

	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	c = max(cnts, key=cv2.contourArea)

	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



# ----------
# show the output image
cv2.imshow("04 Image+Contours", image)
cv2.imshow("03 Distance", D0)
cv2.imshow("02 Thresholded", thresh)
cv2.imshow("01 Shifted", shifted)
cv2.waitKey(0)

