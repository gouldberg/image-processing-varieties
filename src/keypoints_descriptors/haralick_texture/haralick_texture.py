# USAGE
# python classify_texture.py --training training --test testing

import os
os.chdir("C:\\Users\\kswad\\Desktop\\PyImageSearchGurusCourse\\10_06_haralick_texture")


import mahotas
import cv2



# ----------
image_file = "carpet_01.png"
image = cv2.imread(".\\training\\" + image_file)


image_file = "sand.jpg"
image = cv2.imread(image_file)

image.shape


cv2.imshow(image_file, image)
cv2.waitKey(0)



# ----------
# extract Haralick texture features in 4 directions

features = mahotas.features.haralick(image)


# Haralick feautres are derived from the Gray Level Co-occurrence Matrix (GLCM).
# This matrix records how many times two gray-level pixels adjacent to each other appear in an image.
# Then based on this matrix, Haralick proposes 13 values that can be extracted from the GLCM to quantify texture.
# An additional 14 values can be computed; however, they are not often used due to computational instability.

features.shape



# then take the mean of each direction
# This averaging is performed in an attempt to make the feature vector more robust in changes in rotation.
features.mean(axis=0)


