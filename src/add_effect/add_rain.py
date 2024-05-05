
import os
import glob

import cv2
import numpy as np
import PIL.Image


# ----------
# reference:
# https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library


####################################################################################################
# --------------------------------------------------------------------------------------------------
# add rain
# --------------------------------------------------------------------------------------------------

def add_rain(
    img,
    slant,
    drop_length,
    drop_width,
    drop_color,
    blur_value,
    brightness_coefficient,
    rain_drops,
):
    """
    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    Args:
        img (numpy.ndarray): Image.
        slant (int):
        drop_length:
        drop_width:
        drop_color:
        blur_value (int): Rainy view are blurry.
        brightness_coefficient (float): Rainy days are usually shady.
        rain_drops:
    Returns:
        numpy.ndarray: Image.
    """
    # non_rgb_warning(img)
    # ----------
    # input_dtype = img.dtype
    # needs_float = False
    # if input_dtype == np.float32:
    #     img = from_float(img, dtype=np.dtype("uint8"))
    #     needs_float = True
    # elif input_dtype not in (np.uint8, np.float32):
    #     raise ValueError("Unexpected dtype {} for RandomRain augmentation".format(input_dtype))
    # ----------
    image = img.copy()
    for (rain_drop_x0, rain_drop_y0) in rain_drops:
        rain_drop_x1 = rain_drop_x0 + slant
        rain_drop_y1 = rain_drop_y0 + drop_length
        # ----------
        cv2.line(
            image,
            (rain_drop_x0, rain_drop_y0),
            (rain_drop_x1, rain_drop_y1),
            drop_color,
            drop_width,
        )
    # ----------
    image = cv2.blur(image, (blur_value, blur_value))  # rainy view are blurry
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    image_hsv[:, :, 2] *= brightness_coefficient
    # ----------
    image_rgb = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # ----------
    # if needs_float:
    #     image_rgb = to_float(image_rgb, max_value=255)
    # ----------
    return image_rgb


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing/data_augmentation'

image_dir = os.path.join(base_path, 'sample_images')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')

# ----------
index = 0
img = cv2.imread(image_path_list[index])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------------------
# add rain
# --------------------------------------------------------------------------------------------------

slant = 10
drop_length = 10
drop_width = 2
drop_color = (255, 0, 0)
blur_value = 3
brightness_coefficient = 1.0
rain_drops = [(10, 10), (20, 20), (100, 100)]


# ----------
img_rain = add_rain(
    img=img, slant=slant, drop_length=drop_length,
    drop_width=drop_width, drop_color=drop_color,
    blur_value=blur_value, brightness_coefficient=brightness_coefficient,
    rain_drops=rain_drops
    )

PIL.Image.fromarray(img).show()
PIL.Image.fromarray(img_rain).show()


