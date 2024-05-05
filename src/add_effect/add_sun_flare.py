
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
# add 1 sun flare
# --------------------------------------------------------------------------------------------------

def add_sun_flare_simple(img, flare_center_x, flare_center_y, src_radius, src_color, circles):
    """Add sun flare.
    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    Args:
        img (numpy.ndarray):
        flare_center_x (float):
        flare_center_y (float):
        src_radius:
        src_color (int, int, int):
        circles (list):
    Returns:
        numpy.ndarray:
    """
    # non_rgb_warning(img)
	# ----------
    # input_dtype = img.dtype
    # needs_float = False
	# ----------
    # if input_dtype == np.float32:
    #     img = from_float(img, dtype=np.dtype("uint8"))
    #     needs_float = True
    # elif input_dtype not in (np.uint8, np.float32):
    #     raise ValueError("Unexpected dtype {} for RandomSunFlareaugmentation".format(input_dtype))
	# ----------
    overlay = img.copy()
    output = img.copy()
	# ----------
    for (alpha, (x, y), rad3, (r_color, g_color, b_color)) in circles:
        cv2.circle(overlay, (x, y), rad3, (r_color, g_color, b_color), -1)
		# ----------
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	# ----------
    point = (int(flare_center_x), int(flare_center_y))
	# ----------
    overlay = output.copy()
    num_times = src_radius // 10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, src_radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        cv2.addWeighted(overlay, alp, output, 1 - alp, 0, output)
	# ----------
    image_rgb = output
	# ----------
    # if needs_float:
    #     image_rgb = to_float(image_rgb, max_value=255)
	# ----------
    return image_rgb


# --------------------------------------------------------------------------------------------------
# add sun flare
# --------------------------------------------------------------------------------------------------

def flare_source(image,  point,radius, src_color):
    overlay= image.copy()
    output= image.copy()
    num_times=radius//10
    alpha= np.linspace(0.0,1,num= num_times)
    rad= np.linspace(1,radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp=alpha[num_times-i-1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        cv2.addWeighted(overlay, alp, output, 1 - alp ,0, output)
    return output


def add_sun_flare_line(flare_center, angle, imshape):
    x=[]
    y=[]
    i=0
    for rand_x in range(0,imshape[1],10):
        rand_y= math.tan(angle) * (rand_x - flare_center[0]) + flare_center[1]
        x.append(rand_x)
        y.append(2*flare_center[1] - rand_y)
    return x,y


def add_sun_process(image, no_of_flare_circles, flare_center, src_radius, x, y, src_color):
    overlay= image.copy()
    output= image.copy()
    imshape=image.shape
    for i in range(no_of_flare_circles):
        alpha=random.uniform(0.05,0.2)
        r=random.randint(0, len(x)-1)
        rad=random.randint(1, imshape[0]//100-2)
        cv2.circle(overlay,(int(x[r]),int(y[r])), rad*rad*rad, (random.randint(max(src_color[0]-50,0), src_color[0]),random.randint(max(src_color[1]-50,0), src_color[1]),random.randint(max(src_color[2]-50,0), src_color[2])), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)                      
    output= flare_source(output,(int(flare_center[0]),int(flare_center[1])),src_radius,src_color)
    return output


def add_sun_flare(image, flare_center=-1, angle=-1, no_of_flare_circles=8, src_radius=400, src_color=(255,255,255)):
    # verify_image(image)
    # if(angle!=-1):
    #     angle=angle%(2*math.pi)
    # if not(no_of_flare_circles>=0 and no_of_flare_circles<=20):
    #     raise Exception(err_flare_circle_count)
    # if(is_list(image)):
    #     image_RGB=[]
    #     image_list=image
    #     imshape=image_list[0].shape
    #     for img in image_list: 
    #         if(angle==-1):
    #             angle_t=random.uniform(0,2*math.pi)
    #             if angle_t==math.pi/2:
    #                 angle_t=0
    #         else:
    #             angle_t=angle
    #         if flare_center==-1:   
    #             flare_center_t=(random.randint(0,imshape[1]),random.randint(0,imshape[0]//2))
    #         else:
    #             flare_center_t=flare_center
    #         x,y= add_sun_flare_line(flare_center_t,angle_t,imshape)
    #         output= add_sun_process(img, no_of_flare_circles,flare_center_t,src_radius,x,y,src_color)
    #         image_RGB.append(output)
    # else:
    imshape=image.shape
    if(angle==-1):
        angle_t=random.uniform(0, 2 * math.pi)
        if angle_t==math.pi/2:
            angle_t=0
    else:
        angle_t=angle
    if flare_center==-1:
        flare_center_t=(random.randint(0,imshape[1]),random.randint(0,imshape[0]//2))
    else:
        flare_center_t=flare_center
    x,y= add_sun_flare_line(flare_center_t,angle_t,imshape)
    output= add_sun_process(image, no_of_flare_circles, flare_center_t, src_radius, x, y, src_color)
    image_RGB = output
    return image_RGB


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing/data_augmentation'

image_dir = os.path.join(base_path, 'sample_images')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')


index = 0
img = cv2.imread(image_path_list[index])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------------------
# add 1 sun flare
# --------------------------------------------------------------------------------------------------

flare_center_x = 100
flare_center_y = 100
src_radius =200
src_color = (255, 255, 255)

# ----------
alpha = 0.2
x = flare_center_x
y = flare_center_y
rad3 = 90
color = (255, 255, 255)

circles = [(alpha, (x,y), rad3, color)]


# ----------
img_sun_flare = add_sun_flare_simple(img, flare_center_x, flare_center_y, src_radius, src_color, circles)

PIL.Image.fromarray(img).show()
PIL.Image.fromarray(img_sun_flare).show()


# --------------------------------------------------------------------------------------------------
# add random sun flares
# --------------------------------------------------------------------------------------------------

img_sun_flare = add_sun_flare(img)

PIL.Image.fromarray(img).show()
PIL.Image.fromarray(img_sun_flare).show()
