
import os
import glob

import cv2
import numpy as np

import PIL.Image, PIL.ImageDraw, PIL.ImageFilter

import math


# ----------
# reference:
# https://emotionexplorer.blog.fc2.com/blog-entry-190.html


base_path = '/home/kswada/kw/image_processing'
image_dir = os.path.join(base_path, '00_sample_images/illumination_change')



# --------------------------------------------------------------------------------------------------
# Illumination Change:  orange
# --------------------------------------------------------------------------------------------------

src_fname = 'orange.png'
mask_fname = 'orange_mask.png'

src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))

result = cv2.illuminationChange(src, mask, alpha=0.2, beta=0.4)


# ----------
img_to_show = np.hstack([
    cv2.cvtColor(src, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(result, cv2.COLOR_BGR2RGB)    
])

PIL.Image.fromarray(img_to_show).show()


# --------------------------------------------------------------------------------------------------
# Illumination Change:  sumomo
# --------------------------------------------------------------------------------------------------

src_fname = 'sumomo.jpg'
mask_fname = 'sumomo_mask.png'

src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))

result = cv2.illuminationChange(src, mask, alpha=0.5, beta=0.5)


# ----------
img_to_show = np.hstack([
    cv2.cvtColor(src, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(result, cv2.COLOR_BGR2RGB)    
])

PIL.Image.fromarray(img_to_show).show()


# --------------------------------------------------------------------------------------------------
# Illumination Change:  metal
# --------------------------------------------------------------------------------------------------

src_fname = 'metal.jpg'
mask_fname = 'metal_mask.png'

src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))

result = cv2.illuminationChange(src, mask, alpha=0.1, beta=0.5)


# ----------
img_to_show = np.hstack([
    cv2.cvtColor(src, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(result, cv2.COLOR_BGR2RGB)    
])

PIL.Image.fromarray(img_to_show).show()
