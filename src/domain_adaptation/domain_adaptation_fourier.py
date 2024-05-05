
import os
import glob

import numpy as np
import math
import random

import cv2
import PIL.Image


# ----------
# reference:
# https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/domain_adaptation.py
# https://github.com/YanchaoYang/FDA


####################################################################################################
# --------------------------------------------------------------------------------------------------
# FDA: Fourier Domain Adaptation for Semantic Segmentation
#  - Domain adaptation via style transfer made easy using Fourier Transform. FDA needs no deep networks for style transfer, and involves no adversarial training. Below is the diagram of the proposed Fourier Domain Adaptation method:
#  - Step 1: Apply FFT to source and target images.
#  - Step 2: Replace the low frequency part of the source amplitude with that from the target.
#  - Step 3: Apply inverse FFT to the modified source spectrum.
# --------------------------------------------------------------------------------------------------

def fourier_domain_adaptation(img: np.ndarray, target_img: np.ndarray, beta: float) -> np.ndarray:
    """
    Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA
    Args:
        img:  source image
        target_img:  target image for domain adaptation
        beta: coefficient from source paper
    Returns:
        transformed image
    """
    img = np.squeeze(img)
    target_img = np.squeeze(target_img)
    # ----------
    if target_img.shape != img.shape:
        raise ValueError(
            "The source and target images must have the same shape,"
            " but got {} and {} respectively.".format(img.shape, target_img.shape)
        )
    # ----------
    # get fft of both source and target
    fft_src = np.fft.fft2(img.astype(np.float32), axes=(0, 1))
    fft_trg = np.fft.fft2(target_img.astype(np.float32), axes=(0, 1))
    # ----------
    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_trg = np.abs(fft_trg)
    # ----------
    # mutate the amplitude part of source with target
    amplitude_src = np.fft.fftshift(amplitude_src, axes=(0, 1))
    amplitude_trg = np.fft.fftshift(amplitude_trg, axes=(0, 1))
    height, width = amplitude_src.shape[:2]
    border = np.floor(min(height, width) * beta).astype(int)
    center_y, center_x = np.floor([height / 2.0, width / 2.0]).astype(int)
    # ----------
    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1
    # ----------
    amplitude_src[y1:y2, x1:x2] = amplitude_trg[y1:y2, x1:x2]
    amplitude_src = np.fft.ifftshift(amplitude_src, axes=(0, 1))
    # ----------
    # get mutated image
    src_image_transformed = np.fft.ifft2(amplitude_src * np.exp(1j * phase_src), axes=(0, 1))
    src_image_transformed = np.real(src_image_transformed)
    # ----------
    return src_image_transformed


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing/data_augmentation'

image_dir = os.path.join(base_path, 'sample_images')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')


# ----------
# source
# index = 0
# src_img = cv2.imread(image_path_list[index])
src_img = cv2.imread(os.path.join(image_dir, 'source.png'))
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)


# target
# index = 1
# tgt_img = cv2.imread(image_path_list[index])
tgt_img = cv2.imread(os.path.join(image_dir, 'target.png'))
tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
tgt_img = cv2.resize(tgt_img, (src_img.shape[1], src_img.shape[0]))

print(src_img.shape)
print(tgt_img.shape)    


# --------------------------------------------------------------------------------------------------
# fourier domain adaptation
# --------------------------------------------------------------------------------------------------

# coefficient from source paper
# https://github.com/YanchaoYang/FDA
# beta = 0.0:  original image ?
# beta = 1.0:  approaches to target image ?

beta = 0.45

src_img_transformed = fourier_domain_adaptation(
    img = src_img,
    target_img = tgt_img,
    beta = beta
)

src_img_transformed = np.ceil(src_img_transformed).astype('uint8')

PIL.Image.fromarray(src_img).show()
PIL.Image.fromarray(tgt_img).show()

PIL.Image.fromarray(src_img_transformed).show()




