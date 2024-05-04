
import os
import glob

import numpy as np

import cv2
import PIL.Image

from copy import deepcopy

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ----------
# reference:
# https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/domain_adaptation.py


####################################################################################################
# --------------------------------------------------------------------------------------------------
# domain adapter
# --------------------------------------------------------------------------------------------------

# class TransformerInterface(Protocol):
#     @abc.abstractmethod
#     def inverse_transform(self, X: np.ndarray) -> np.ndarray:
#         ...
#     @abc.abstractmethod
#     def fit(self, X: np.ndarray, y=None):
#         ...
#     @abc.abstractmethod
#     def transform(self, X: np.ndarray, y=None) -> np.ndarray:
#         ...

class DomainAdapter:
    def __init__(self, transformer, ref_img, color_conversions=(None, None)):
        self.color_in, self.color_out = color_conversions
        self.source_transformer = deepcopy(transformer)
        self.target_transformer = transformer
        self.target_transformer.fit(self.flatten(ref_img))
    # ----------
    def to_colorspace(self, img):
        if self.color_in is None:
            return img
        return cv2.cvtColor(img, self.color_in)
    # ----------
    def from_colorspace(self, img):
        if self.color_out is None:
            return img
        return cv2.cvtColor(img.astype('uint8'), self.color_out)
    # ----------
    def flatten(self, img):
        img = self.to_colorspace(img)
        img = img.astype('float32') / 255.
        return img.reshape(-1, 3)
    # ----------
    def reconstruct(self, pixels, h, w):
        pixels = (np.clip(pixels, 0, 1) * 255).astype('uint8')
        return self.from_colorspace(pixels.reshape(h, w, 3))
    # ----------
    @staticmethod
    def _pca_sign(x):
        return np.sign(np.trace(x.components_))
    # ----------
    def __call__(self, image: np.ndarray):
        h, w, _ = image.shape
        pixels = self.flatten(image)
        self.source_transformer.fit(pixels)
        # ----------
        if self.target_transformer.__class__ in (PCA,):
            # dirty hack to make sure colors are not inverted
            if self._pca_sign(self.target_transformer) != self._pca_sign(self.source_transformer):
                self.target_transformer.components_ *= -1
        # ----------
        representation = self.source_transformer.transform(pixels)
        result = self.target_transformer.inverse_transform(representation)
        return self.reconstruct(result, h, w)
    

# --------------------------------------------------------------------------------------------------
# pixel distribution adaptation
# --------------------------------------------------------------------------------------------------

def adapt_pixel_distribution(img, ref, transform_type='pca', weight=0.5):
    initial_type = img.dtype
    transformer = {"pca": PCA, "standard": StandardScaler, "minmax": MinMaxScaler}[transform_type]()
    adapter = DomainAdapter(transformer=transformer, ref_img=ref)
    result = adapter(img).astype("float32")
    blended = (img.astype("float32") * (1 - weight) + result * weight).astype(initial_type)
    return blended


# --------------------------------------------------------------------------------------------------
# pixel distribution adaptation:  simple pca by np.linalg.eig
# --------------------------------------------------------------------------------------------------

def adapt_pixel_distribution_simple_pca(src_img, tgt_img, weight=0.5):
    initial_type = src_img.dtype
    tgt_pixels = tgt_img.astype('float32') / 255.
    tgt_pixels = tgt_pixels.reshape(-1, 3)
    # ----------
    tgt_mean = np.nanmean(tgt_pixels, axis=0)
    tgt_scale = np.nanstd(tgt_pixels, axis=0)
    tgt_S = np.cov((tgt_pixels - tgt_mean).T, bias=1)
    # tgt_S = np.cov(((tgt_pixels - tgt_mean)/tgt_scale).T, bias=1)
    # ----------
    # tgt_eig = np.linalg.eig(tgt_S)[0]
    tgt_eigvec = np.linalg.eig(tgt_S)[1]
    # tgt_idx = np.argsort(tgt_eig)[::-1]
    # tgt_eig = tgt_eig[tgt_idx]
    # tgt_eigvec = tgt_eigvec[tgt_idx]
    # ----------
    h, w, _ = src_img.shape
    src_pixels = src_img.astype('float32') / 255.
    src_pixels = src_pixels.reshape(-1, 3)
    # ----------
    src_mean = np.mean(src_pixels, axis=0)
    src_scale = np.nanstd(src_pixels, axis=0)
    src_S = np.cov((src_pixels - src_mean).T, bias=1)
    # src_S = np.cov(((src_pixels - src_mean)/src_scale).T, bias=1)
    # ----------
    # src_eig = np.linalg.eig(src_S)[0]
    src_eigvec = np.linalg.eig(src_S)[1]
    # src_idx = np.argsort(src_eig)[::-1]
    # src_eig = src_eig[src_idx]
    # src_eigvec = src_eigvec[src_idx]
    # ----------
    if np.sign(np.trace(src_eigvec)) != np.sign(np.trace(tgt_eigvec)):
        tgt_eigvec = -1 * tgt_eigvec
    representation = np.dot(src_pixels - src_mean, src_eigvec.T)
    result = np.dot(representation, tgt_eigvec) + tgt_mean
    pixels = (np.clip(result, 0, 1) * 255).astype('uint8')
    result = pixels.reshape(h, w, 3).astype('float32')
    src_img_transformed = (src_img.astype("float32") * (1 - weight) + result * weight).astype(initial_type)
    return src_img_transformed


# --------------------------------------------------------------------------------------------------
# pixel distribution adaptation:  simple pca by np.linalg.svd
# --------------------------------------------------------------------------------------------------

# Vt: components_
def adapt_pixel_distribution_simple_pca2(src_img, tgt_img, weight=0.5):
    initial_type = src_img.dtype
    tgt_pixels = tgt_img.astype('float32') / 255.
    tgt_pixels = tgt_pixels.reshape(-1, 3)
    # ----------
    tgt_mean = np.mean(tgt_pixels, axis=0)
    tgt_scale = np.nanstd(tgt_pixels, axis=0)
    tgt_pixels -= tgt_mean
    # tgt_pixels /= tgt_scale
    tU, tS, tVt = np.linalg.svd(tgt_pixels, full_matrices=False)
    # t_max_abs_cols = np.argmax(np.abs(tU), axis=0)
    # t_signs = np.sign(tU[t_max_abs_cols, range(tU.shape[1])])
    # tU *= t_signs
    # tVt *= t_signs[:, np.newaxis]
    # ----------
    h, w, _ = src_img.shape
    src_pixels = src_img.astype('float32') / 255.
    src_pixels = src_pixels.reshape(-1, 3)
    # ----------
    src_mean = np.mean(src_pixels, axis=0)
    src_scale = np.nanstd(src_pixels, axis=0)
    src_pixels -= src_mean
    # src_pixels /= src_scale
    sU, sS, sVt = np.linalg.svd(src_pixels, full_matrices=False)
    # s_max_abs_cols = np.argmax(np.abs(sU), axis=0)
    # s_signs = np.sign(sU[s_max_abs_cols, range(sU.shape[1])])
    # sU *= s_signs
    # sVt *= s_signs[:, np.newaxis]
    # ----------
    if np.sign(np.trace(sVt)) != np.sign(np.trace(tVt)):
        tVt = -1 * tVt
    # ----------
    representation = np.dot(src_pixels, sVt.T)
    result = np.dot(representation, tVt) + tgt_mean
    pixels = (np.clip(result, 0, 1) * 255).astype('uint8')
    result = pixels.reshape(h, w, 3).astype('float32')
    src_img_transformed = (src_img.astype("float32") * (1 - weight) + result * weight).astype(initial_type)
    return src_img_transformed


####################################################################################################
# --------------------------------------------------------------------------------------------------
# load image
# --------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing'

image_dir = os.path.join(base_path, '00_sample_images')

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*jpg')))
print(f'num of images: {len(image_path_list)}')


# ----------
src_img = cv2.imread(os.path.join(image_dir, 'road_day.png'))
# tgt_img = cv2.imread(os.path.join(image_dir, 'road_dark.png'))
# tgt_img = cv2.imread(os.path.join(image_dir, 'strong_light.jpg'))

# src_img = cv2.imread(os.path.join(image_dir, 'empire_state_cloudy.png'))
# tgt_img = cv2.imread(os.path.join(image_dir, 'white_balance/red/girl.jpg'))

# src_img = cv2.imread(os.path.join(image_dir, 'bedroom.jpg'))
# tgt_img = cv2.imread(os.path.join(image_dir, 'strong_light.jpg'))
# tgt_img = cv2.imread(os.path.join(image_dir, 'white_balance/red/girl.jpg'))
tgt_img = cv2.imread(os.path.join(image_dir, 'white_balance/blue/sunny_day_building.jpg'))


src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
tgt_img = cv2.resize(tgt_img, (src_img.shape[1], src_img.shape[0]))

print(src_img.shape)
print(tgt_img.shape)    


# --------------------------------------------------------------------------------------------------
# pixel distribution adaptation
# --------------------------------------------------------------------------------------------------

# transform type is 'pca', 'standard' or 'minxmax'

transform_type = 'pca'
# transform_type = 'standard'
# transform_type = 'minmax'

weight = 0.5

src_img_transformed = adapt_pixel_distribution(
    img = src_img,
    ref = tgt_img,
    transform_type = transform_type,
    weight = weight
)


src_img_transformed2 = adapt_pixel_distribution_simple_pca(
    src_img = src_img,
    tgt_img = tgt_img,
    weight = weight
)

src_img_transformed3 = adapt_pixel_distribution_simple_pca2(
    src_img = src_img,
    tgt_img = tgt_img,
    weight = weight
)

img_to_show = np.hstack([src_img, tgt_img, src_img_transformed, src_img_transformed2, src_img_transformed3])

PIL.Image.fromarray(img_to_show).show()


