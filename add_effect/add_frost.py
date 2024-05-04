
import os
import glob

import cv2
import numpy as np
import PIL.Image


# ----------
# reference:
# https://github.com/CrazyVertigo/imagecorruptions/blob/master/imagecorruptions/corruptions.py


####################################################################################################
# --------------------------------------------------------------------------------------------------
# add frost
# --------------------------------------------------------------------------------------------------

def add_frost(x, resource_file, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    # ----------
    # idx = np.random.randint(5)
    # filename = [resource_filename(__name__, './frost/frost1.png'),
    #             resource_filename(__name__, './frost/frost2.png'),
    #             resource_filename(__name__, './frost/frost3.png'),
    #             resource_filename(__name__, './frost/frost4.jpg'),
    #             resource_filename(__name__, './frost/frost5.jpg'),
    #             resource_filename(__name__, './frost/frost6.jpg')][idx]
    frost = cv2.imread(resource_file)
    frost_shape = frost.shape
    x_shape = np.array(x).shape
    # ----------
    # resize the frost image so it fits to the image dimensions
    scaling_factor = 1
    if frost_shape[0] >= x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = 1
    elif frost_shape[0] < x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = x_shape[0] / frost_shape[0]
    elif frost_shape[0] >= x_shape[0] and frost_shape[1] < x_shape[1]:
        scaling_factor = x_shape[1] / frost_shape[1]
    elif frost_shape[0] < x_shape[0] and frost_shape[1] < x_shape[1]: 
        # If both dims are too small, pick the bigger scaling factor
        scaling_factor_0 = x_shape[0] / frost_shape[0]
        scaling_factor_1 = x_shape[1] / frost_shape[1]
        scaling_factor = np.maximum(scaling_factor_0, scaling_factor_1)
    # ----------
    scaling_factor *= 1.1
    new_shape = (int(np.ceil(frost_shape[1] * scaling_factor)),
                 int(np.ceil(frost_shape[0] * scaling_factor)))
    frost_rescaled = cv2.resize(frost, dsize=new_shape,
                                interpolation=cv2.INTER_CUBIC)
    # ----------
    # randomly crop
    x_start, y_start = np.random.randint(0, frost_rescaled.shape[0] - x_shape[0]), \
        np.random.randint(0, frost_rescaled.shape[1] - x_shape[1])
    # ----------
    if len(x_shape) < 3 or x_shape[2] < 3:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                         y_start:y_start + x_shape[1]]
        frost_rescaled = np.dot(frost_rescaled[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                         y_start:y_start + x_shape[1]][..., [2, 1, 0]]
    return np.clip(c[0] * np.array(x) + c[1] * frost_rescaled, 0, 255)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load frost source file
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir = os.path.join(base_path, '00_sample_images')

resource_frost_img_paths = sorted(glob.glob(os.path.join(image_dir, 'resource_frost/*')))



####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

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
# add frost
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
    # add frost
    # --------------------------------------------------------------------------------------------------
    img_frost_list = []
    for i in range(len(resource_frost_img_paths)):
        img_frost = add_frost(img, resource_frost_img_paths[i])
        img_frost_rgb = cv2.cvtColor(img_frost.astype('uint8'), cv2.COLOR_BGR2RGB)
        img_frost_list.append(img_frost_rgb)
        # ----------
    for j in range(len(resource_frost_img_paths)):
        if j == 0:
            img_to_show = np.hstack([img_rgb, img_frost_list[j]])
        else:
            img_to_show = np.hstack([img_to_show, img_frost_list[j]])
    PIL.Image.fromarray(img_to_show.astype('uint8')).show()

