# USAGE
# python basic_connected_components.py --image license_plate.png

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")
import argparse
import cv2
import os


# -----------------------------------------------------------------------------------------------
# set arguments
# -----------------------------------------------------------------------------------------------

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-c", "--connectivity", type=int, default=4,
# 	help="connectivity for connected component analysis")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20210222_opencv-connected-components"


# -----------------------------------------------------------------------------------------------
# load input image and preprocess by grayscaling and threshold
# -----------------------------------------------------------------------------------------------

img_file = 'license_plate.png'

image = cv2.imread(os.path.join(base_path, img_file))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# binary_inv + otsu
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


cv2.imshow("Original", image)
cv2.waitKey(0)

cv2.imshow("Thresholded", thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()


# -----------------------------------------------------------------------------------------------
# apply connected component analysis to the thresholded image
# -----------------------------------------------------------------------------------------------

connectivity = 4
# connectivity = 8

# cv2.connectedComponentsWithAlgorithm:
#  - implements faster, more efficient algorithms for connected component analysis.

# call to cv2.connectedComponentsWithStats here.
output = cv2.connectedComponentsWithStats(
	thresh, connectivity, cv2.CV_32S)

(numLabels, labels, stats, centroids) = output


# The total number of unique labels (i.e., number of total components) that were detected
print(numLabels)

# A mask named labels has the same spatial dimensions as our input thresh image.
# For each location in labels, we have an integer ID value that corresponds
# to the connected component where the pixel belongs.
print(labels)
print(labels.shape)

# Statistics on each connected component, including the bounding box coordinates
# and area (in pixels).
print(stats)
print(stats.shape)

# the centroids (i.e., center) (x, y)-coordinates of each connected component.
print(centroids)
print(centroids.shape)



# -----------------------------------------------------------------------------------------------
# loop over the number of unique connected component labels
# -----------------------------------------------------------------------------------------------

for i in range(0, numLabels):
	# if this is the first component then we examining the
	# *background* (typically we would just ignore this
	# component in our loop)
	if i == 0:
		text = "examining component {}/{} (background)".format(
			i + 1, numLabels)

	# otherwise, we are examining an actual connected component
	else:
		text = "examining component {}/{}".format( i + 1, numLabels)

	# print a status message update for the current connected
	# component
	print("[INFO] {}".format(text))

	# extract the connected component statistics and centroid for
	# the current label
	x = stats[i, cv2.CC_STAT_LEFT]
	y = stats[i, cv2.CC_STAT_TOP]
	w = stats[i, cv2.CC_STAT_WIDTH]
	h = stats[i, cv2.CC_STAT_HEIGHT]
	area = stats[i, cv2.CC_STAT_AREA]
	(cX, cY) = centroids[i]

	# clone our original image (so we can draw on it) and then draw
	# a bounding box surrounding the connected component along with
	# a circle corresponding to the centroid
	output = image.copy()
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
	cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

	# construct a mask for the current connected component by
	# finding a pixels in the labels array that have the current
	# connected component ID
	componentMask = (labels == i).astype("uint8") * 255

	# show our output image and connected component mask
	cv2.imshow("Output", output)
	cv2.imshow("Connected Component", componentMask)
	cv2.waitKey(0)


cv2.destroyAllWindows()

