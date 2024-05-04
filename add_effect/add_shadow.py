
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
# add shadows
# --------------------------------------------------------------------------------------------------

def add_shadow_simple(img, vertices_list):
    """Add shadows to the image.
    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    Args:
        img (numpy.ndarray):
        vertices_list (list):
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
    #     raise ValueError("Unexpected dtype {} for RandomShadow augmentation".format(input_dtype))
    # ----------
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(img)
    # ----------
    # adding all shadow polygons on empty mask, single 255 denotes only red channel
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255)
    # ----------
    # if red channel is hot, image's "Lightness" channel's brightness is lowered
    red_max_value_ind = mask[:, :, 0] == 255
    image_hls[:, :, 1][red_max_value_ind] = image_hls[:, :, 1][red_max_value_ind] * 0.5
    # ----------
    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    # ----------
    # if needs_float:
    #     image_rgb = to_float(image_rgb, max_value=255)
    # ----------
    return image_rgb


# --------------------------------------------------------------------------------------------------
# add shadows
# --------------------------------------------------------------------------------------------------

def generate_shadow_coordinates(imshape, no_of_shadows, rectangular_roi, shadow_dimension):
    vertices_list=[]
    x1=rectangular_roi[0]
    y1=rectangular_roi[1]
    x2=rectangular_roi[2]
    y2=rectangular_roi[3]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(shadow_dimension): ## Dimensionality of the shadow polygon
            vertex.append((random.randint(x1, x2),random.randint(y1, y2)))
        vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices 
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices


def shadow_process(image,no_of_shadows,x1,y1,x2,y2, shadow_dimension):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    mask = np.zeros_like(image) 
    imshape = image.shape
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows,(x1,y1,x2,y2), shadow_dimension) #3 getting list of shadow vertices
    for vertices in vertices_list: 
        cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5   ## if red channel is hot, image's "Lightness" channel's brightness is lowered 
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB


def add_shadow(image, no_of_shadows=1, rectangular_roi=(-1,-1,-1,-1), shadow_dimension=5):
    ## ROI:(top-left x1,y1, bottom-right x2,y2), shadow_dimension=no. of sides of polygon generated
    # verify_image(image)
    # if not(is_numeric(no_of_shadows) and no_of_shadows>=1 and no_of_shadows<=10):
    #     raise Exception(err_shadow_count)
    # if not(is_numeric(shadow_dimension) and shadow_dimension>=3 and shadow_dimension<=10):
    #     raise Exception(err_shadow_dimension)
    # if is_tuple(rectangular_roi) and is_numeric_list_or_tuple(rectangular_roi) and len(rectangular_roi)==4:
    x1=rectangular_roi[0]
    y1=rectangular_roi[1]
    x2=rectangular_roi[2]
    y2=rectangular_roi[3]
    # else:
    #     raise Exception(err_invalid_rectangular_roi)
    # if rectangular_roi==(-1,-1,-1,-1):
    #     x1=0
    #     if(is_numpy_array(image)):
    #         y1=image.shape[0]//2
    #         x2=image.shape[1]
    #         y2=image.shape[0]
    #     else:
    #         y1=image[0].shape[0]//2
    #         x2=image[0].shape[1]
    #         y2=image[0].shape[0]
    # elif x1==-1 or y1==-1 or x2==-1 or y2==-1 or x2<=x1 or y2<=y1:
    #     raise Exception(err_invalid_rectangular_roi)
    # if(is_list(image)):
    #     image_RGB=[]
    #     image_list=image
    #     for img in image_list:
    #         output=shadow_process(img,no_of_shadows,x1,y1,x2,y2, shadow_dimension)
    #         image_RGB.append(output)
    # else:
    output=shadow_process(image, no_of_shadows, x1, y1, x2, y2, shadow_dimension)
    image_RGB = output
    return image_RGB


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir = os.path.join(base_path, '00_sample_images')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')

# ----------
index = 0
img = cv2.imread(image_path_list[index])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------------------
# add shadow:  simple version
# --------------------------------------------------------------------------------------------------

vertices_list = [[np.array([(300, 200), (220, 300), (300, 340), (340, 220)]),]]

vertices_list = [[
    np.array([(300, 200), (220, 300), (300, 340), (340, 220)]),
    np.array([(200, 100), (210, 200), (250, 240), (330, 120)])
    ]]


# ----------
img_shadow = add_shadow_simple(img, vertices_list)

PIL.Image.fromarray(img).show()
PIL.Image.fromarray(img_shadow).show()


# --------------------------------------------------------------------------------------------------
# add shadows
# --------------------------------------------------------------------------------------------------

no_of_shadows = 10
rectangular_roi = (100, 100, 600, 600)
shadow_dimension = 4

img_shadow = add_shadow(
    image=img,
    no_of_shadows=no_of_shadows,
    rectangular_roi=rectangular_roi,
    shadow_dimension=shadow_dimension
    )

PIL.Image.fromarray(img).show()
PIL.Image.fromarray(img_shadow).show()



