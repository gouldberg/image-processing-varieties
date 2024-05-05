# USAGE
# python sobel.py --image bricks.png
import os
os.chdir("/media/kswada/MyFiles/PyImageSearchGurusCourse/01_09_01_gradients/")
print(os.getcwd())

# import the necessary packages
import argparse
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())


# load the image, convert it to grayscale, and display the original
# image
# image = cv2.imread("bricks.png")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)


# compute gradients along the X and Y axis, respectively
gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)


# the `gX` and `gY` images are now of the floating point data type,
# so we need to take care to convert them back a to unsigned 8-bit
# integer representation so other OpenCV functions can utilize them
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)


# combine the sobel X and Y representations into a single image
sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)


# show our output images
cv2.imshow("Sobel X", gX)
cv2.imshow("Sobel Y", gY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)



# ----------
# test
#import numpy as np
#gY = 253 - 67
#gX = 224 - 231
#mag = np.sqrt((gX ** 2) + (gY ** 2))
#orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
#print(orientation)
#
#
#
## ----------
#img = np.array([[44,67,96],[231,184,224],[51,253,36]]).astype("uint8")
#gX = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0)
#gY = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1)
#
#mag = np.sqrt((gX ** 2) + (gY ** 2))
#orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
#print(orientation)
