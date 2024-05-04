# USAGE
# python index_features.py --dataset ~/Desktop/ukbench_sample --features-db output/features.hdf5
from __future__ import print_function


import os
os.chdir("/media/kswada/MyFiles/PyImageSearchGurusCourse/03_05_extracting_keypoints_and_local_invariant_descriptors")
print(os.getcwd())


# import the necessary packages
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.indexer import FeatureIndexer
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features-db", required=True,
	help="Path to where the features database will be stored")
ap.add_argument("-a", "--approx-images", type=int, default=500,
	help="Approximate # of images in the dataset")
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000,
	help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())


# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline


# ----------
# Changed !!!
# detector = FeatureDetector_create("SURF")
# descriptor = DescriptorExtractor_create("RootSIFT")
detector = cv2.xfeatures2d.SURF_create()
descriptor = cv2.xfeatures2d.SIFT_create()


dad = DetectAndDescribe(detector, descriptor)


# initialize the feature indexer
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],
	maxBufferSize=args["max_buffer_size"], verbose=True)


# loop over the images in the dataset
for (i, imagePath) in enumerate(paths.list_images(args["dataset"])):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

	# extract the image filename (i.e. the unique image ID) from the image
	# path, then load the image itself
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=320)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# describe the image
	(kps, descs) = dad.describe(image)

	# if either the keypoints or descriptors are None, then ignore the image
	if kps is None or descs is None:
		continue

	# index the features
	fi.add(filename, kps, descs)


# finish the indexing process
fi.finish()



# ----------
# test
# =============================================================================
# import h5py
# db = h5py.File("./output/features.hdf5", mode="r")
# list(db.keys())
# 
# db["image_ids"].shape
# db["index"].shape
# db["features"].shape
# 
# 
# for image in range(db["image_ids"].shape[0]):
#     if(db["image_ids"][image] == "ukbench03098.jpg"):
#         print(image)
# 
# imageID = db["image_ids"][218]
# imageID
# 
# (start, end) = db["index"][218]
# (start, end)
# end - start
# 
# rows = db["features"][start:end]
# rows.shape
# kps = rows[:, :2]
# descs = rows[:, 2:]
# 
# kps.shape
# 
# descs.shape
# 
# 
# =============================================================================
