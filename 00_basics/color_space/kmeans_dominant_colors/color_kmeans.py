# USAGE
# python color_kmeans.py --image images/jp.png --clusters 3

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

import os


# -------------------------------------------------------------------------------------
# centroid_histogram(), plot_colors()
# -------------------------------------------------------------------------------------

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins=numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist


def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype="uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
					  color.astype("uint8").tolist(), -1)
		startX = endX

	# return the bar chart
	return bar


# -------------------------------------------------------------------------------------
# set arguments
# -------------------------------------------------------------------------------------

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# ap.add_argument("-c", "--clusters", required = True, type = int,
# 	help = "# of clusters")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20140526_python-kmeans-dominant-colors"


# -------------------------------------------------------------------------------------
# load image
# -------------------------------------------------------------------------------------

img_file = os.path.join(base_path, "images\\batman.png")
img_file = os.path.join(base_path, "images\\jp.png")
img_file = os.path.join(base_path, "images\\thematrix.png")


image = cv2.imread(img_file)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ----------
# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)


# -------------------------------------------------------------------------------------
# k-means clustering
# -------------------------------------------------------------------------------------

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))


# cluster the pixel intensities
n_clus = 3

clt = KMeans(n_clusters = n_clus)

clt.fit(image)



# -------------------------------------------------------------------------------------
# display color bar
# -------------------------------------------------------------------------------------

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color

hist = centroid_histogram(clt)

bar = plot_colors(hist, clt.cluster_centers_)


# ----------
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
