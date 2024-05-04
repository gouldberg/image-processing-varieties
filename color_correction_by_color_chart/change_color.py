
import os
import glob

import cv2
import numpy as np

import PIL.Image, PIL.ImageDraw, PIL.ImageFilter

import math


# ----------
# reference:
# http://amroamroamro.github.io/mexopencv/opencv/cloning_demo.html


base_path = '/home/kswada/kw/image_processing'
image_dir = os.path.join(base_path, '00_sample_images/color_change')



# --------------------------------------------------------------------------------------------------
# Color Change:  flower
# --------------------------------------------------------------------------------------------------

src_fname = 'flower.png'
mask_fname = 'flower_mask.png'

src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))

result = cv2.colorChange(src, mask, red_mul=1.5, green_mul=0.5, blue_mul=0.5)


# ----------
img_to_show = np.hstack([
    cv2.cvtColor(src, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(result, cv2.COLOR_BGR2RGB)    
])

PIL.Image.fromarray(img_to_show).show()


# --------------------------------------------------------------------------------------------------
# Color Change:  yasai
# --------------------------------------------------------------------------------------------------

src_fname = 'yasai.jpg'
mask_fname = 'yasai_mask.png'

src = cv2.imread(os.path.join(image_dir, src_fname))
mask = cv2.imread(os.path.join(image_dir, mask_fname))

result1 = cv2.colorChange(src, mask, red_mul=0.5, green_mul=2.5, blue_mul=2.5)
result2 = cv2.colorChange(src, mask, red_mul=2.5, green_mul=0.5, blue_mul=2.5)
result3 = cv2.colorChange(src, mask, red_mul=2.5, green_mul=2.5, blue_mul=0.5)


# ----------
img_to_show = np.hstack([
    cv2.cvtColor(src, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(result1, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(result2, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(result3, cv2.COLOR_BGR2RGB)  
])

PIL.Image.fromarray(img_to_show).show()




