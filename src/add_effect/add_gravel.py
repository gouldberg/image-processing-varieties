
import os
import glob

import numpy as np
import math
import random

import cv2
import PIL.Image


# ----------
# reference:
# https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library


####################################################################################################
# --------------------------------------------------------------------------------------------------
# add gravel simple version
# --------------------------------------------------------------------------------------------------

def add_gravel_simple(img: np.ndarray, gravels: list):
    """Add gravel to the image.
    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    Args:
        img (numpy.ndarray): image to add gravel to
        gravels (list): list of gravel parameters. (float, float, float, float):
            (top-left x, top-left y, bottom-right x, bottom right y)
    Returns:
        numpy.ndarray:
    """
    # non_rgb_warning(img)
    # input_dtype = img.dtype
    # needs_float = False
    # ----------
    # if input_dtype == np.float32:
    #     img = from_float(img, dtype=np.dtype("uint8"))
    #     needs_float = True
    # elif input_dtype not in (np.uint8, np.float32):
    #     raise ValueError("Unexpected dtype {} for AddGravel augmentation".format(input_dtype))
    # ----------
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # ----------
    for gravel in gravels:
        y1, y2, x1, x2, sat = gravel
        image_hls[x1:x2, y1:y2, 1] = sat
    # ----------
    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    # ----------
    # if needs_float:
    #     image_rgb = to_float(image_rgb, max_value=255)
    # ----------
    return image_rgb


# --------------------------------------------------------------------------------------------------
# add gravel
# --------------------------------------------------------------------------------------------------

def hls(image,src='RGB'):
    # verify_image(image)
    # if(is_list(image)):
    #     image_HLS=[]
    #     image_list=image
    #     for img in image_list:
    #         eval('image_HLS.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2HLS))')
    # else:
    image_HLS = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HLS)')
    return image_HLS


def rgb(image, src='BGR'):
    # verify_image(image)
    # if(is_list(image)):
    #     image_RGB=[]
    #     image_list=image
    #     for img in image_list:
    #         eval('image_RGB.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB))')
    # else:
    image_RGB= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)')
    return image_RGB


def generate_gravel_patch(rectangular_roi):
    x1=rectangular_roi[0]
    y1=rectangular_roi[1]
    x2=rectangular_roi[2]
    y2=rectangular_roi[3] 
    gravels=[]
    area= abs((x2-x1)*(y2-y1))
    for i in range((int)(area//10)):
        x= np.random.randint(x1,x2)
        y= np.random.randint(y1,y2)
        gravels.append((x,y))
    return gravels


def gravel_process(image,x1,x2,y1,y2,no_of_patches):
    x=image.shape[1]
    y=image.shape[0]
    rectangular_roi_default=[]
    for i in range(no_of_patches):
        xx1=random.randint(x1, x2)
        xx2=random.randint(x1, xx1)
        yy1=random.randint(y1, y2)
        yy2=random.randint(y1, yy1)
        rectangular_roi_default.append((xx2,yy2,min(xx1,xx2+200),min(yy1,yy2+30)))
    img_hls=hls(image)
    for roi in rectangular_roi_default:
        gravels= generate_gravel_patch(roi)
        for gravel in gravels:
            x=gravel[0]
            y=gravel[1]
            r=random.randint(1, 4)
            r1=random.randint(0, 255)
            img_hls[max(y-r,0):min(y+r,y),max(x-r,0):min(x+r,x),1]=r1
    image_RGB= rgb(img_hls,'hls') 
    return image_RGB


def add_gravel(image, rectangular_roi=(-1,-1,-1,-1), no_of_patches=8):
    # verify_image(image)
    # if is_tuple(rectangular_roi) and is_numeric_list_or_tuple(rectangular_roi) and len(rectangular_roi)==4:
    x1=rectangular_roi[0]
    y1=rectangular_roi[1]
    x2=rectangular_roi[2]
    y2=rectangular_roi[3]
    # else:
    #     raise Exception(err_invalid_rectangular_roi)
    # if rectangular_roi==(-1,-1,-1,-1):
    #     if(is_numpy_array(image)):
    #         x1=0
    #         y1=int(image.shape[0]*3/4)
    #         x2=image.shape[1]
    #         y2=image.shape[0]
    #     else:
    #         x1=0
    #         y1=int(image[0].shape[0]*3/4)
    #         x2=image[0].shape[1]
    #         y2=image[0].shape[0]
    # elif x1==-1 or y1==-1 or x2==-1 or y2==-1 or x2<=x1 or y2<=y1:
    #     raise Exception(err_invalid_rectangular_roi)
    color=[0,255]  
    # if(is_list(image)):
    #     image_RGB=[]
    #     image_list=image
    #     for img in image_list:
    #         output= gravel_process(img,x1,x2,y1,y2,no_of_patches)
    #         image_RGB.append(output)
    # else:
    output= gravel_process(image,x1,x2,y1,y2,no_of_patches)
    image_RGB= output    
    return image_RGB


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
# add gravel simple version
# --------------------------------------------------------------------------------------------------

gravels = [
    (100,200,100,200,100)
]


img_gravel = add_gravel_simple(img, gravels)

# PIL.Image.fromarray(img).show()
PIL.Image.fromarray(img_gravel).show()


# --------------------------------------------------------------------------------------------------
# add gravel
# --------------------------------------------------------------------------------------------------

rectangular_roi = (100, 100, 600, 600)
no_of_patches = 10

img_gravel = add_gravel(
    image=img,
    rectangular_roi=rectangular_roi,
    no_of_patches=no_of_patches
    )

# PIL.Image.fromarray(img).show()
PIL.Image.fromarray(img_gravel).show()

