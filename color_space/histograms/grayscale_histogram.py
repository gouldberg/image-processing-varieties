# USAGE
# python grayscale_histogram.py --image beach.png
import os
os.chdir("/media/kswada/MyFiles/PyImageSearchGurusCourse/01_12_histograms")
print(os.getcwd())


# import the necessary packages
from matplotlib import pyplot as plt
import argparse
import cv2


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())


# load the image, convert it to grayscale, and show it
image = cv2.imread(args["image"])
# image = cv2.imread("horseshoe_bend.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)


# construct a grayscale histogram
# A grayscale image has only one channel, so we have a value of [0]  for channels.
# We don’t have a mask, so we set the mask value to None.
# We will use 256 bins in our histogram, and the possible values range from 0 to 255.
hist = cv2.calcHist([image], [0], None, [256], [0, 256])


# plot the histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])


# normalize the histogram
hist /= hist.sum()


# plot the normalized histogram
plt.figure()
plt.title("Grayscale Histogram (Normalized)")
plt.xlabel("Bins")
plt.ylabel("% of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()