# USAGE
# python single_template_matching.py --image images/coke_bottle.png --template images/coke_logo.png

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import argparse
import cv2

import os


# -----------------------------------------------------------------------------------------------
# set arguments
# -----------------------------------------------------------------------------------------------

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True,
# 	help="path to input image where we'll apply template matching")
# ap.add_argument("-t", "--template", type=str, required=True,
# 	help="path to template image")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20210322_opencv-template-matching"


# -----------------------------------------------------------------------------------------------
# load image and template
# -----------------------------------------------------------------------------------------------

img_file = os.path.join(base_path, 'images\\8_diamonds.png')
tmplt = os.path.join(base_path, 'images\\diamonds_template.png')

img_file = os.path.join(base_path, 'images\\coke_bottle.png.png')
tmplt = os.path.join(base_path, 'images\\coke_logo.png.png')

img_file = os.path.join(base_path, 'images\\coke_bottle_rotated.png')
tmplt = os.path.join(base_path, 'images\\coke_logo.png.png')


print("[INFO] loading images...")
image = cv2.imread(img_file)
template = cv2.imread(tmplt)
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

print("[INFO] performing template matching...")

result = cv2.matchTemplate(imageGray, templateGray,
	cv2.TM_CCOEFF_NORMED)


# ----------
# takes correlation result and returns a 4-tuple which includes the minimum correlation value,
# the (x,y)-coordinate of the minimum value, and the (x,y)-coordinate of the maximum value, respectively.

(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

# determine the starting and ending (x, y)-coordinates of the
# bounding box
(startX, startY) = maxLoc
endX = startX + template.shape[1]
endY = startY + template.shape[0]

# draw the bounding box on the image
cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)

