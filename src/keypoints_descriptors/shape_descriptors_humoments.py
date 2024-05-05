# USAGE
# python humoments.py

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

import cv2

import os



# ------------------------------------------------------------------------------------------
# load image
# ------------------------------------------------------------------------------------------

base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20141027_opencv-shape-descriptors"


image = cv2.imread(os.path.join(base_path, "diamond.png"))

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# ------------------------------------------------------------------------------------------
# Extract Hu Moments
# ------------------------------------------------------------------------------------------

# extract Hu Moments from the image -- this list of numbers
# is our 'feature vector' used to quantify the shape of the
# object in our image

features = cv2.HuMoments(cv2.moments(image)).flatten()

print(features)


# --> 7 features



# ----------
cv2.imshow("image", image)
cv2.waitKey(0)

