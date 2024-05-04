# USAGE
# python max_filter.py --image images/horseshoe_bend_02.jpg

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")


import numpy as np
import argparse
import cv2

import os



# -------------------------------------------------------------------------------------------
# max_rgb_filter()
# -------------------------------------------------------------------------------------------

def max_rgb_filter(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)

	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	R[R < M] = 0
	G[G < M] = 0
	B[B < M] = 0

	# merge the channels back together and return the image
	return cv2.merge([B, G, R])



# -------------------------------------------------------------------------------------------
# set arguments
# -------------------------------------------------------------------------------------------

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20150928_max-filter"



# -------------------------------------------------------------------------------------------
# load image
# -------------------------------------------------------------------------------------------

img_file = os.path.join(base_path, "images\\angels_landing.png")
img_file = os.path.join(base_path, "images\\antelope_canyon.png")
img_file = os.path.join(base_path, "images\\grand_canyon.png")
img_file = os.path.join(base_path, "images\\horseshoe_bend_01.png")
img_file = os.path.join(base_path, "images\\horseshoe_bend_02.png")


image = cv2.imread(img_file)



# -------------------------------------------------------------------------------------------
# Max RGB filter
# -------------------------------------------------------------------------------------------

filtered = max_rgb_filter(image)



# ----------
cv2.imshow("Images", np.hstack([image, filtered]))
cv2.waitKey(0)



# -------------------------------------------------------------------------------------------
# step by step
# -------------------------------------------------------------------------------------------

(B, G, R) = cv2.split(image)

print(image.shape)
print(B.shape)


# ----------
M = np.maximum(np.maximum(R, G), B)

print(M.shape)


# ----------
R[R < M] = 0
G[G < M] = 0
B[B < M] = 0


filtered = cv2.merge([B, G, R])

print(filtered.shape)
