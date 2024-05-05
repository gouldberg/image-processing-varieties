# USAGE
# python recognize.py --training images/training --testing images/testing

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2

import os



# ----------------------------------------------------------------------------------
# class LocalBinaryPatterns()
# ----------------------------------------------------------------------------------

from skimage import feature
import numpy as np

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist



# ----------------------------------------------------------------------------------
# set arguments
# ----------------------------------------------------------------------------------

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--training", required=True,
# 	help="path to the training images")
# ap.add_argument("-e", "--testing", required=True,
# 	help="path to the tesitng images")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20151207_local-binary-patterns"



# ----------------------------------------------------------------------------------
# extract feature by Local Binary Patterns
# ----------------------------------------------------------------------------------

img_train_path = os.path.join(base_path, "images\\training")


# initialize the local binary patterns descriptor along with
# the data and label lists

# radius = 8: pattern surrounding the central pixel
# The radius of the circle r, which allows us to account for different scales

# num_points = 24: the number of points along the outer radius
# The number of points p in a circularly symmetric neighborhood to consider
# (thus removing relying on a square neighborhood)

desc = LocalBinaryPatterns(24, 8)

data = []

labels = []



# ----------
# loop over the training images
for imagePath in paths.list_images(img_train_path):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split(os.path.sep)[-2])
	data.append(hist)


# ----------
# 4 textures * 4 images = 16 images
print(len(data))
print(len(labels))


# there are p + 1 uniform patterns.
# The final dimensionality of the histogram is thus p + 2,
# where the added entry tabulates all patterns that are not uniform
# = 24 + 2 = 26
print(len(data[0]))
print(data[0])



# ----------------------------------------------------------------------------------
# train SVM
# ----------------------------------------------------------------------------------

# train a Linear SVM on the data

model = LinearSVC(C=100.0, random_state=42)

model.fit(data, labels)



# ----------------------------------------------------------------------------------
# test images
# ----------------------------------------------------------------------------------

img_test_path = os.path.join(base_path, "images\\testing")


# loop over the testing images

for imagePath in paths.list_images(img_test_path):
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict(hist.reshape(1, -1))

	# display the image and the prediction
	cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)



# ----------------------------------------------------------------------------------
# feature.local_binary_pattern():  step by step
# ----------------------------------------------------------------------------------

img = np.array([5,4,2,2,1,3,5,8,1,3,2,5,4,1,2,4,3,7,2,7,1,4,4,2,6]).reshape(5, 5)

print(img)

numPoints = 8
radius = 4

lbp = feature.local_binary_pattern(img, P=numPoints, R=radius, method="uniform")

print(lbp)


(hist, _) = np.histogram(lbp.ravel(),
						 bins=np.arange(0, numPoints + 3),
						 range=(0, numPoints + 2))


# normalize the histogram
hist = hist.astype("float")

eps = 1e-7
hist /= (hist.sum() + eps)

