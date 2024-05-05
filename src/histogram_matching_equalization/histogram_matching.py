
import os
import glob

import numpy as np
import math
import random

import cv2
import PIL.Image
import matplotlib.pyplot as plt

from skimage import exposure


# Histogram matching can best be thought of as a “transformation.”
# Our goal is to take an input image
# (the “source”) and update its pixel intensities such that the distribution of the input image histogram
# matches the distribution of a reference image.
# While the input image’s actual contents do not change, the pixel distribution does,
# thereby adjusting the illumination and contrast of the input image based on the distribution
# of the reference image.
# Applying histogram matching allows us to obtain interesting aesthetic results
# (as we’ll see later in this tutorial).
# Additionally, we can use histogram matching as a form of basic color correction/color constancy,


####################################################################################################
# -----------------------------------------------------------------------------------------------
# histogram matching by numpy
# -----------------------------------------------------------------------------------------------

def histogram_matching(src, ref):
    src_lookup = src.reshape(-1)
    src_counts = np.bincount(src_lookup)
    tmpl_counts = np.bincount(ref.reshape(-1))
    # ----------
    # omit values where the count was 0
    tmpl_values = np.nonzero(tmpl_counts)[0]
    tmpl_counts = tmpl_counts[tmpl_values]
    # ----------
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / src.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / ref.size
    # ----------
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(src.shape)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir = os.path.join(base_path, '00_sample_images')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')


# ----------
# img = cv2.imread(os.path.join(image_dir, 'empire_state_cloudy.png'))
# ref = cv2.imread(os.path.join(image_dir, 'empire_state_sunset.png'))

img = cv2.imread(os.path.join(image_dir, 'road_day.png'))
ref = cv2.imread(os.path.join(image_dir, 'road_dark.png'))

img = cv2.imread(os.path.join(image_dir, 'strong_light.jpg'))
ref = cv2.imread(os.path.join(image_dir, 'road_day.png'))


height = img.shape[0]
width = img.shape[1]
width_resize = 640

scale = width_resize / width

height_resize = int(height * scale)
width_resize = int(width * scale)

img = cv2.resize(img, (width_resize, height_resize))
ref = cv2.resize(ref, (width_resize, height_resize))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------------------------
# histogram matching by skimage exposure.match_histograms
# -----------------------------------------------------------------------------------------------

multi = True

matched = np.zeros(img.shape)
matched[...,0] = exposure.match_histograms(img[..., 0], ref[..., 0])
matched[...,1] = exposure.match_histograms(img[..., 1], ref[..., 1])
matched[...,2] = exposure.match_histograms(img[..., 2], ref[..., 2])

img_to_show = np.hstack([img, ref, matched])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()


# -----------------------------------------------------------------------------------------------
# histogram matching by numpy
# -----------------------------------------------------------------------------------------------

matched = np.zeros(img.shape)
matched[...,0] = histogram_matching(img[..., 0], ref[..., 0])
matched[...,1] = histogram_matching(img[..., 1], ref[..., 1])
matched[...,2] = histogram_matching(img[..., 2], ref[..., 2])

img_to_show = np.hstack([img, ref, matched])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()


# ----------
# only Y in Y/U/V

img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
ref_yuv = cv2.cvtColor(ref, cv2.COLOR_RGB2YUV)

matched = np.zeros(img.shape)
matched[...,0] = histogram_matching(img[..., 0], ref[..., 0])
matched[...,1] = img[..., 1]
matched[...,2] = img[..., 2]

matched = cv2.cvtColor(matched, cv2.COLOR_YUV2RGB).astype('uint8')

img_to_show = np.hstack([img, ref, matched])

PIL.Image.fromarray(img_to_show.astype('uint8')).show()


# -----------------------------------------------------------------------------------------------
# show the matching
# -----------------------------------------------------------------------------------------------

(fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

# loop over our source image, reference image, and output matched
# image
for (i, image) in enumerate((img, ref, matched.astype('uint8'))):
	# convert the image from BGR to RGB channel ordering
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# loop over the names of the channels in RGB order
	for (j, color) in enumerate(("red", "green", "blue")):
		# compute a histogram for the current channel and plot it
		(hist, bins) = exposure.histogram(image[..., j],
			source_range="dtype")
		axs[j, i].plot(bins, hist / hist.max())
		# compute the cumulative distribution function for the
		# current channel and plot it
		(cdf, bins) = exposure.cumulative_distribution(image[..., j])
		axs[j, i].plot(bins, cdf)
		# set the y-axis label of the current plot to be the name
		# of the current color channel
		axs[j, 0].set_ylabel(color)

# set the axes titles
axs[0, 0].set_title("Source")
axs[0, 1].set_title("Reference")
axs[0, 2].set_title("Matched")

# display the output plots
plt.tight_layout()
plt.show()

