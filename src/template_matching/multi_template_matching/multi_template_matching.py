# USAGE
# python multi_template_matching.py --image images/8_diamonds.png --template images/diamonds_template.png

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2

import os


# -----------------------------------------------------------------------------------------------
# set arguments
# -----------------------------------------------------------------------------------------------

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True,
# 	help="path to input image where we'll apply template matching")
# ap.add_argument("-t", "--template", type=str, required=True,
# 	help="path to template image")
# ap.add_argument("-b", "--threshold", type=float, default=0.8,
# 	help="threshold for multi-template matching")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20210329_multi-template-matching"


# -----------------------------------------------------------------------------------------------
# load images and template
# -----------------------------------------------------------------------------------------------

img_file = os.path.join(base_path, 'images\\8_diamonds.png')
tmplt = os.path.join(base_path, 'images\\diamonds_template.png')

# load the input image and template image from disk, then grab the
# template image spatial dimensions
print("[INFO] loading images...")
image = cv2.imread(img_file)
template = cv2.imread(tmplt)
(tH, tW) = template.shape[:2]

# display the  image and template to our screen
cv2.imshow("Image", image)
cv2.imshow("Template", template)
cv2.waitKey(0)


# -----------------------------------------------------------------------------------------------
# convert both the image and template to grayscale
# -----------------------------------------------------------------------------------------------

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


# -----------------------------------------------------------------------------------------------
# perform template matching
# -----------------------------------------------------------------------------------------------

# values in the range[0.8, 0.95] typically work the best
thresh = 0.8

print("[INFO] performing template matching...")
result = cv2.matchTemplate(imageGray, templateGray,
	cv2.TM_CCOEFF_NORMED)

# find all locations in the result map where the matched value is
# greater than the threshold, then clone our original image so we
# can draw on it
(yCoords, xCoords) = np.where(result >= thresh)
clone = image.copy()
print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))


# ----------
# loop over our starting (x, y)-coordinates
for (x, y) in zip(xCoords, yCoords):
	# draw the bounding box on the image
	cv2.rectangle(clone, (x, y), (x + tW, y + tH),
		(255, 0, 0), 3)

# show our output image *before* applying non-maxima suppression
cv2.imshow("Before NMS", clone)
cv2.waitKey(0)

# initialize our list of rectangles
rects = []

# loop over the starting (x, y)-coordinates again
for (x, y) in zip(xCoords, yCoords):
	# update our list of rectangles
	rects.append((x, y, x + tW, y + tH))

# apply non-maxima suppression to the rectangles
pick = non_max_suppression(np.array(rects))
print("[INFO] {} matched locations *after* NMS".format(len(pick)))

# loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
	# draw the bounding box on the image
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(255, 0, 0), 3)

# show the output image
cv2.imshow("After NMS", image)
cv2.waitKey(0)

