
import os
import glob

import cv2
import numpy as np
import PIL.Image


# ----------
# reference:
# https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
# https://github.com/CrazyVertigo/imagecorruptions/blob/master/imagecorruptions/corruptions.py


####################################################################################################
# --------------------------------------------------------------------------------------------------
# add fog
# --------------------------------------------------------------------------------------------------

# this requires parameters ...

def add_fog(img, fog_coef, alpha_coef, haze_list):
    """Add fog to the image.
    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    Args:
        img (numpy.ndarray): Image.
        fog_coef (float): Fog coefficient.
        alpha_coef (float): Alpha coefficient.
        haze_list (list):
    Returns:
        numpy.ndarray: Image.
    """
    # non_rgb_warning(img)
    # input_dtype = img.dtype
    # needs_float = False
    # if input_dtype == np.float32:
    #     img = from_float(img, dtype=np.dtype("uint8"))
    #     needs_float = True
    # elif input_dtype not in (np.uint8, np.float32):
    #     raise ValueError("Unexpected dtype {} for RandomFog augmentation".format(input_dtype))
    # ----------
    width = img.shape[1]
    hw = max(int(width // 3 * fog_coef), 10)
    for haze_points in haze_list:
        x, y = haze_points
        overlay = img.copy()
        output = img.copy()
        alpha = alpha_coef * fog_coef
        rad = hw // 2
        point = (x + hw // 2, y + hw // 2)
        cv2.circle(overlay, point, int(rad), (255, 255, 255), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        img = output.copy()
    image_rgb = cv2.blur(img, (hw // 10, hw // 10))
    # ----------
    # if needs_float:
    #     image_rgb = to_float(image_rgb, max_value=255)
    # ----------
    return image_rgb


# --------------------------------------------------------------------------------------------------
# add fog2:
# https://github.com/CrazyVertigo/imagecorruptions/blob/master/imagecorruptions/corruptions.py
# --------------------------------------------------------------------------------------------------

# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100
    # ----------
    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)
    # ----------
    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)
    # ----------
    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
        0:mapsize:stepsize] = wibbledmean(ttsum)
    # ----------
    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    # ----------
    maparray -= maparray.min()
    return maparray / maparray.max()


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def add_fog2(x, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    # ----------
    shape = np.array(x).shape
    max_side = np.max(shape)
    map_size = next_power_of_2(int(max_side))
    x = np.array(x) / 255.
    max_val = x.max()
    x_shape = np.array(x).shape
    if len(x_shape) < 3 or x_shape[2] < 3:
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[
                    :shape[0], :shape[1]]
    else:
        x += c[0] * \
             plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0], :shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir = os.path.join(base_path, '00_sample_images')

# image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
# print(f'num of images: {len(image_path_list)}')


# ----------
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/blue/*')))
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/yellow/*')))
img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/red/*')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, 'white_balance/*.png')))

img_file_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
    sorted(glob.glob(os.path.join(image_dir, '*.png')))


# --------------------------------------------------------------------------------------------------
# add fog
# --------------------------------------------------------------------------------------------------

for i in range(len(img_file_list)):
    img_file = img_file_list[i]
    img = cv2.imread(img_file)
    height = img.shape[0]
    width = img.shape[1]
    width_resize = 640
    scale = width_resize / width
    height_resize = int(height * scale)
    width_resize = int(width * scale)
    img = cv2.resize(img, (width_resize, height_resize))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # --------------------------------------------------------------------------------------------------
    # add fog2
    # --------------------------------------------------------------------------------------------------
    for i in range(5):
        img_fog_rgb = add_fog2(img, severity=i+1).astype('uint8')
        if i == 0:
            img_to_show = np.hstack([img_rgb, img_fog_rgb])
        else:
            img_to_show = np.hstack([img_to_show, img_fog_rgb])
    PIL.Image.fromarray(img_to_show.astype('uint8')).show()


