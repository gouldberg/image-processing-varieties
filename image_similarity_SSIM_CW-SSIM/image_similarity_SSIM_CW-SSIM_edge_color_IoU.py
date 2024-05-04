
import os
import numpy as np

import cv2

import PIL.Image
from PIL import ImageOps
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity
from skimage.color import deltaE_ciede2000
from skimage import color

from scipy import signal
import scipy.ndimage as ndi
from scipy.spatial import distance as dist

import albumentations as A

base_path = '/home/kswada/kw/image_processing'



####################################################################################################
# --------------------------------------------------------------------------------------------------
# SSIM and CW-SSIM
# https://github.com/jterrace/pyssim/blob/master/ssim/ssimlib.py
# --------------------------------------------------------------------------------------------------

def convolve_gaussian_2d(image, gaussian_kernel_1d):
    """Convolve 2d gaussian."""
    result = ndi.correlate1d(image, gaussian_kernel_1d, axis=0)
    return ndi.correlate1d(result, gaussian_kernel_1d, axis=1)


def get_gaussian_kernel(gaussian_kernel_width=11, gaussian_kernel_sigma=1.5):
    """Generate a gaussian kernel."""
    # 1D Gaussian kernel definition
    gaussian_kernel_1d = np.arange(0, gaussian_kernel_width, 1.)
    gaussian_kernel_1d -= gaussian_kernel_width / 2
    gaussian_kernel_1d = np.exp(-0.5 * gaussian_kernel_1d**2 / gaussian_kernel_sigma**2)
    return gaussian_kernel_1d / np.sum(gaussian_kernel_1d)


def to_grayscale(img):
    """Convert PIL image to numpy grayscale array and numpy alpha array.

    Args:
      img (PIL.Image): PIL Image object.

    Returns:
      (gray, alpha): both numpy arrays.
    """
    gray = np.asarray(PIL.ImageOps.grayscale(img)).astype(float)

    imbands = img.getbands()
    alpha = None
    if 'A' in imbands:
        alpha = np.asarray(img.split()[-1]).astype(float)

    return gray, alpha


# ----------
class SSIMImage(object):
    """Wraps a PIL Image object with SSIM state.

    Attributes:
      img: Original PIL Image.
      img_gray: grayscale Image.
      img_gray_squared: squared img_gray.
      img_gray_mu: img_gray convolved with gaussian kernel.
      img_gray_mu_squared: squared img_gray_mu.
      img_gray_sigma_squared: img_gray convolved with gaussian kernel -
                              img_gray_mu_squared.
    """
    def __init__(self, img, gaussian_kernel_1d=None, size=None):
        """Create an SSIMImage.

        Args:
          img (str or PIL.Image): PIL Image object or file name.
          gaussian_kernel_1d (np.ndarray, optional): Gaussian kernel
          that was generated with utils.get_gaussian_kernel is used
          to precompute common objects for SSIM computation
          size (tuple, optional): New image size to resize image to.
        """
        # Use existing or create a new PIL.Image
        self.img = PIL.Image.open(img)

        # Resize image if size is defined and different
        # from original image
        if size and size != self.img.size:
            self.img = self.img.resize(size, PIL.Image.ANTIALIAS)

        # Set the size of the image
        self.size = self.img.size

        # If gaussian kernel is defined we create
        # common SSIM objects
        if gaussian_kernel_1d is not None:

            self.gaussian_kernel_1d = gaussian_kernel_1d

            # np.array of grayscale and alpha image
            self.img_gray, self.img_alpha = to_grayscale(self.img)
            if self.img_alpha is not None:
                self.img_gray[self.img_alpha == 255] = 0

            # Squared grayscale
            self.img_gray_squared = self.img_gray ** 2

            # Convolve grayscale image with gaussian
            self.img_gray_mu = convolve_gaussian_2d(
                self.img_gray, self.gaussian_kernel_1d)

            # Squared mu
            self.img_gray_mu_squared = self.img_gray_mu ** 2

            # Convolve squared grayscale with gaussian
            self.img_gray_sigma_squared = convolve_gaussian_2d(
                self.img_gray_squared, self.gaussian_kernel_1d)

            # Substract squared mu
            self.img_gray_sigma_squared -= self.img_gray_mu_squared

        # If we don't define gaussian kernel, we create
        # common CW-SSIM objects
        else:
            # Grayscale PIL.Image
            self.img_gray = ImageOps.grayscale(self.img)

# ----------
class SSIM(object):
    """Computes SSIM between two images."""
    def __init__(self, img, gaussian_kernel_1d=None, size=None,
                 l=255, k_1=0.01, k_2=0.03, k=0.01):
        """Create an SSIM object.

        Args:
          img (str or PIL.Image): Reference image to compare other images to.
          l, k_1, k_2 (float): SSIM configuration variables.
          k (float): CW-SSIM configuration variable (default 0.01)
          gaussian_kernel_1d (np.ndarray, optional): Gaussian kernel
          that was generated with utils.get_gaussian_kernel is used
          to precompute common objects for SSIM computation
          size (tuple, optional): resize the image to the tuple size
        """
        self.k = k
        # Set k1,k2 & c1,c2 to depend on L (width of color map).
        self.c_1 = (k_1 * l) ** 2
        self.c_2 = (k_2 * l) ** 2
        self.gaussian_kernel_1d = gaussian_kernel_1d
        self.img = SSIMImage(img, gaussian_kernel_1d, size)

    def ssim_value(self, target):
        """Compute the SSIM value from the reference image to the target image.

        Args:
          target (str or PIL.Image): Input image to compare the reference image
          to. This may be a PIL Image object or, to save time, an SSIMImage
          object (e.g. the img member of another SSIM object).

        Returns:
          Computed SSIM float value.
        """
        # Performance boost if handed a compatible SSIMImage object.
        if not isinstance(target, SSIMImage) \
          or not np.array_equal(self.gaussian_kernel_1d,
                                target.gaussian_kernel_1d):
            target = SSIMImage(target, self.gaussian_kernel_1d, self.img.size)

        img_mat_12 = self.img.img_gray * target.img_gray
        img_mat_sigma_12 = convolve_gaussian_2d(
            img_mat_12, self.gaussian_kernel_1d)
        img_mat_mu_12 = self.img.img_gray_mu * target.img_gray_mu
        img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

        # Numerator of SSIM
        num_ssim = ((2 * img_mat_mu_12 + self.c_1) *
                    (2 * img_mat_sigma_12 + self.c_2))

        # Denominator of SSIM
        den_ssim = (
            (self.img.img_gray_mu_squared + target.img_gray_mu_squared +
             self.c_1) *
            (self.img.img_gray_sigma_squared +
             target.img_gray_sigma_squared + self.c_2))

        ssim_map = num_ssim / den_ssim
        index = np.average(ssim_map)
        return index

    def cw_ssim_value(self, target, width=30):
        """Compute the complex wavelet SSIM (CW-SSIM) value from the reference
        image to the target image.

        Args:
          target (str or PIL.Image): Input image to compare the reference image
          to. This may be a PIL Image object or, to save time, an SSIMImage
          object (e.g. the img member of another SSIM object).
          width: width for the wavelet convolution (default: 30)

        Returns:
          Computed CW-SSIM float value.
        """
        if not isinstance(target, SSIMImage):
            target = SSIMImage(target, size=self.img.size)

        # Define a width for the wavelet convolution
        widths = np.arange(1, width+1)

        # Use the image data as arrays
        sig1 = np.asarray(self.img.img_gray.getdata())
        sig2 = np.asarray(target.img_gray.getdata())

        # Convolution
        cwtmatr1 = signal.cwt(sig1, signal.ricker, widths)
        cwtmatr2 = signal.cwt(sig2, signal.ricker, widths)

        # Compute the first term
        c1c2 = np.multiply(abs(cwtmatr1), abs(cwtmatr2))
        c1_2 = np.square(abs(cwtmatr1))
        c2_2 = np.square(abs(cwtmatr2))
        num_ssim_1 = 2 * np.sum(c1c2, axis=0) + self.k
        den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + self.k

        # Compute the second term
        c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
        num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + self.k
        den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + self.k

        # Construct the result
        ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)

        # Average the per pixel results
        index = np.average(ssim_map)
        return index


# --------------------------------------------------------------------------------------------------
# KW function:  SSIM and CW-SSIM
# --------------------------------------------------------------------------------------------------

def ssim_val_func(img_gray1, img_gray2, gaussian_kernel_id, l=255, k_1=0.01, k_2=0.03, k=0.01):
    # ----------
    img_gray1 = img_gray1.astype(float)
    img_gray2 = img_gray2.astype(float)
    # ----------
    c_1 = (k_1 * l) ** 2
    c_2 = (k_2 * l) ** 2
    # ----------
    # Squared grayscale
    img_gray_squared1 = img_gray1 ** 2
    img_gray_squared2 = img_gray2 ** 2
    # ----------
    # Convolve grayscale image with gaussian
    img_gray_mu1 = convolve_gaussian_2d(img_gray1, gaussian_kernel_1d)
    img_gray_mu2 = convolve_gaussian_2d(img_gray2, gaussian_kernel_1d)
    # ----------
    # Squared mu
    img_gray_mu_squared1 = img_gray_mu1 ** 2
    img_gray_mu_squared2 = img_gray_mu2 ** 2
    # ----------
    # Convolve squared grayscale with gaussian
    img_gray_sigma_squared1 = convolve_gaussian_2d(img_gray_squared1, gaussian_kernel_1d)
    img_gray_sigma_squared2 = convolve_gaussian_2d(img_gray_squared2, gaussian_kernel_1d)
    # ----------
    # Subtract squared mu
    img_gray_sigma_squared1 -= img_gray_mu_squared1
    img_gray_sigma_squared2 -= img_gray_mu_squared2
    # ----------
    img_mat_12 = img_gray1 * img_gray2
    img_mat_sigma_12 = convolve_gaussian_2d(img_mat_12, gaussian_kernel_1d)
    img_mat_mu_12 = img_gray_mu1 * img_gray_mu2
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12
    # ----------
    # Numerator of SSIM
    num_ssim = ((2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2))
    # Denominator of SSIM
    den_ssim = (
            (img_gray_mu_squared1 + img_gray_mu_squared2 + c_1) *
            (img_gray_sigma_squared1 + img_gray_sigma_squared2 + c_2))
    # ----------
    return np.average(num_ssim / den_ssim)


def cwssim_val_func(img_gray1, img_gray2, k=0.01, cw_width=30):
    img_gray1 = img_gray1.astype(float)
    img_gray2 = img_gray2.astype(float)
    # ----------
    # Define a width for the wavelet convolution
    widths = np.arange(1, cw_width + 1)
    # ----------
    sig1 = img_gray1.reshape(-1)
    sig2 = img_gray2.reshape(-1)
    # ----------
    # Convolution
    cwtmatr1 = signal.cwt(sig1, signal.ricker, widths)
    cwtmatr2 = signal.cwt(sig2, signal.ricker, widths)
    # ----------
    # Compute the first term
    c1c2 = np.multiply(abs(cwtmatr1), abs(cwtmatr2))
    c1_2 = np.square(abs(cwtmatr1))
    c2_2 = np.square(abs(cwtmatr2))
    num_ssim_1 = 2 * np.sum(c1c2, axis=0) + k
    den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + k
    # Compute the second term
    c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
    num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + k
    den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + k
    # ----------
    return np.average((num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2))


def ssim_mch_val_func(img1, img2, gaussian_kernel_id, l=255, k_1=0.01, k_2=0.03, k=0.01, mode='mean'):
    img1_ch0, img1_ch1, img1_ch2 = cv2.split(img1)
    img2_ch0, img2_ch1, img2_ch2 = cv2.split(img2)
    img1_ch0 = img1_ch0.astype(float)
    img1_ch1 = img1_ch1.astype(float)
    img1_ch2 = img1_ch2.astype(float)
    img2_ch0 = img2_ch0.astype(float)
    img2_ch1 = img2_ch1.astype(float)
    img2_ch2 = img2_ch2.astype(float)
    # ----------
    c_1 = (k_1 * l) ** 2
    c_2 = (k_2 * l) ** 2
    # ----------
    # Squared grayscale
    img_gray_squared1_0 = img1_ch0 ** 2
    img_gray_squared1_1 = img1_ch1 ** 2
    img_gray_squared1_2 = img1_ch2 ** 2
    img_gray_squared2_0 = img2_ch0 ** 2
    img_gray_squared2_1 = img2_ch1 ** 2
    img_gray_squared2_2 = img2_ch2 ** 2
    # ----------
    # Convolve grayscale image with gaussian
    img_gray_mu1_0 = convolve_gaussian_2d(img1_ch0, gaussian_kernel_1d)
    img_gray_mu1_1 = convolve_gaussian_2d(img1_ch1, gaussian_kernel_1d)
    img_gray_mu1_2 = convolve_gaussian_2d(img1_ch2, gaussian_kernel_1d)
    img_gray_mu2_0 = convolve_gaussian_2d(img2_ch0, gaussian_kernel_1d)
    img_gray_mu2_1 = convolve_gaussian_2d(img2_ch1, gaussian_kernel_1d)
    img_gray_mu2_2 = convolve_gaussian_2d(img2_ch2, gaussian_kernel_1d)
    # ----------
    # Squared mu
    img_gray_mu_squared1_0 = img_gray_mu1_0 ** 2
    img_gray_mu_squared1_1 = img_gray_mu1_1 ** 2
    img_gray_mu_squared1_2 = img_gray_mu1_2 ** 2
    img_gray_mu_squared2_0 = img_gray_mu2_0 ** 2
    img_gray_mu_squared2_1 = img_gray_mu2_1 ** 2
    img_gray_mu_squared2_2 = img_gray_mu2_2 ** 2
    # ----------
    # Convolve squared grayscale with gaussian
    img_gray_sigma_squared1_0 = convolve_gaussian_2d(img_gray_squared1_0, gaussian_kernel_1d)
    img_gray_sigma_squared1_1 = convolve_gaussian_2d(img_gray_squared1_1, gaussian_kernel_1d)
    img_gray_sigma_squared1_2 = convolve_gaussian_2d(img_gray_squared1_2, gaussian_kernel_1d)
    img_gray_sigma_squared2_0 = convolve_gaussian_2d(img_gray_squared2_0, gaussian_kernel_1d)
    img_gray_sigma_squared2_1 = convolve_gaussian_2d(img_gray_squared2_1, gaussian_kernel_1d)
    img_gray_sigma_squared2_2 = convolve_gaussian_2d(img_gray_squared2_2, gaussian_kernel_1d)
    # ----------
    # Subtract squared mu
    img_gray_sigma_squared1_0 -= img_gray_mu_squared1_0
    img_gray_sigma_squared1_1 -= img_gray_mu_squared1_1
    img_gray_sigma_squared1_2 -= img_gray_mu_squared1_2
    img_gray_sigma_squared2_0 -= img_gray_mu_squared2_0
    img_gray_sigma_squared2_1 -= img_gray_mu_squared2_1
    img_gray_sigma_squared2_2 -= img_gray_mu_squared2_2
    # ----------
    img_mat_12_0 = img1_ch0 * img2_ch0
    img_mat_12_1 = img1_ch1 * img2_ch1
    img_mat_12_2 = img1_ch2 * img2_ch2
    img_mat_sigma_12_0 = convolve_gaussian_2d(img_mat_12_0, gaussian_kernel_1d)
    img_mat_sigma_12_1 = convolve_gaussian_2d(img_mat_12_1, gaussian_kernel_1d)
    img_mat_sigma_12_2 = convolve_gaussian_2d(img_mat_12_2, gaussian_kernel_1d)
    img_mat_mu_12_0 = img_gray_mu1_0 * img_gray_mu2_0
    img_mat_mu_12_1 = img_gray_mu1_1 * img_gray_mu2_1
    img_mat_mu_12_2 = img_gray_mu1_2 * img_gray_mu2_2
    img_mat_sigma_12_0 = img_mat_sigma_12_0 - img_mat_mu_12_0
    img_mat_sigma_12_1 = img_mat_sigma_12_1 - img_mat_mu_12_1
    img_mat_sigma_12_2 = img_mat_sigma_12_2 - img_mat_mu_12_2
    # ----------
    # Numerator of SSIM
    num_ssim_0 = ((2 * img_mat_mu_12_0 + c_1) * (2 * img_mat_sigma_12_0 + c_2))
    num_ssim_1 = ((2 * img_mat_mu_12_1 + c_1) * (2 * img_mat_sigma_12_1 + c_2))
    num_ssim_2 = ((2 * img_mat_mu_12_2 + c_1) * (2 * img_mat_sigma_12_2 + c_2))
    # Denominator of SSIM
    den_ssim_0 = (
            (img_gray_mu_squared1_0 + img_gray_mu_squared2_0 + c_1) *
            (img_gray_sigma_squared1_0 + img_gray_sigma_squared2_0 + c_2))
    den_ssim_1 = (
            (img_gray_mu_squared1_1 + img_gray_mu_squared2_1 + c_1) *
            (img_gray_sigma_squared1_1 + img_gray_sigma_squared2_1 + c_2))
    den_ssim_2 = (
            (img_gray_mu_squared1_2 + img_gray_mu_squared2_2 + c_1) *
            (img_gray_sigma_squared1_2 + img_gray_sigma_squared2_2 + c_2))
    # ----------
    val0 = np.average(num_ssim_0 / den_ssim_0)
    val1 = np.average(num_ssim_1 / den_ssim_1)
    val2 = np.average(num_ssim_2 / den_ssim_2)
    if mode == 'max':
        return np.max([val0, val1, val2])
    else:
        return np.average([val0, val1, val2])


def cwssim_mch_val_func(img1, img2, k=0.01, cw_width=30, mode='mean'):
    img1_ch0, img1_ch1, img1_ch2 = cv2.split(img1)
    img2_ch0, img2_ch1, img2_ch2 = cv2.split(img2)
    img1_ch0 = img1_ch0.astype(float)
    img1_ch1 = img1_ch1.astype(float)
    img1_ch2 = img1_ch2.astype(float)
    img2_ch0 = img2_ch0.astype(float)
    img2_ch1 = img2_ch1.astype(float)
    img2_ch2 = img2_ch2.astype(float)
    # ----------
    # Define a width for the wavelet convolution
    widths = np.arange(1, cw_width + 1)
    # ----------
    sig1_0 = img1_ch0.reshape(-1)
    sig1_1 = img1_ch1.reshape(-1)
    sig1_2 = img1_ch2.reshape(-1)
    sig2_0 = img2_ch0.reshape(-1)
    sig2_1 = img2_ch1.reshape(-1)
    sig2_2 = img2_ch2.reshape(-1)
    # ----------
    # Convolution
    cwtmatr1_0 = signal.cwt(sig1_0, signal.ricker, widths)
    cwtmatr1_1 = signal.cwt(sig1_1, signal.ricker, widths)
    cwtmatr1_2 = signal.cwt(sig1_2, signal.ricker, widths)
    cwtmatr2_0 = signal.cwt(sig2_0, signal.ricker, widths)
    cwtmatr2_1 = signal.cwt(sig2_1, signal.ricker, widths)
    cwtmatr2_2 = signal.cwt(sig2_2, signal.ricker, widths)
    # ----------
    # Compute the first term
    c1c2_0 = np.multiply(abs(cwtmatr1_0), abs(cwtmatr2_0))
    c1c2_1 = np.multiply(abs(cwtmatr1_1), abs(cwtmatr2_1))
    c1c2_2 = np.multiply(abs(cwtmatr1_2), abs(cwtmatr2_2))
    c1_2_0 = np.square(abs(cwtmatr1_0))
    c2_2_0 = np.square(abs(cwtmatr2_0))
    c1_2_1 = np.square(abs(cwtmatr1_1))
    c2_2_1 = np.square(abs(cwtmatr2_1))
    c1_2_2 = np.square(abs(cwtmatr1_2))
    c2_2_2 = np.square(abs(cwtmatr2_2))
    num_ssim_1_0 = 2 * np.sum(c1c2_0, axis=0) + k
    num_ssim_1_1 = 2 * np.sum(c1c2_1, axis=0) + k
    num_ssim_1_2 = 2 * np.sum(c1c2_2, axis=0) + k
    den_ssim_1_0 = np.sum(c1_2_0, axis=0) + np.sum(c2_2_0, axis=0) + k
    den_ssim_1_1 = np.sum(c1_2_1, axis=0) + np.sum(c2_2_1, axis=0) + k
    den_ssim_1_2 = np.sum(c1_2_2, axis=0) + np.sum(c2_2_2, axis=0) + k
    # Compute the second term
    c1c2_conj_0 = np.multiply(cwtmatr1_0, np.conjugate(cwtmatr2_0))
    c1c2_conj_1 = np.multiply(cwtmatr1_1, np.conjugate(cwtmatr2_1))
    c1c2_conj_2 = np.multiply(cwtmatr1_2, np.conjugate(cwtmatr2_2))
    num_ssim_2_0 = 2 * np.abs(np.sum(c1c2_conj_0, axis=0)) + k
    num_ssim_2_1 = 2 * np.abs(np.sum(c1c2_conj_1, axis=0)) + k
    num_ssim_2_2 = 2 * np.abs(np.sum(c1c2_conj_2, axis=0)) + k
    den_ssim_2_0 = 2 * np.sum(np.abs(c1c2_conj_0), axis=0) + k
    den_ssim_2_1 = 2 * np.sum(np.abs(c1c2_conj_1), axis=0) + k
    den_ssim_2_2 = 2 * np.sum(np.abs(c1c2_conj_2), axis=0) + k
    # ----------
    val0 = np.average((num_ssim_1_0 / den_ssim_1_0) * (num_ssim_2_0 / den_ssim_2_0))
    val1 = np.average((num_ssim_1_1 / den_ssim_1_1) * (num_ssim_2_1 / den_ssim_2_1))
    val2 = np.average((num_ssim_1_2 / den_ssim_1_2) * (num_ssim_2_2 / den_ssim_2_2))
    if mode == 'max':
        return np.max([val0, val1, val2])
    else:
        return np.average([val0, val1, val2])


# --------------------------------------------------------------------------------------------------
# color histgram distance
# --------------------------------------------------------------------------------------------------

def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])


def calc_hist_dist(img1, img2, channels=[0, 1, 2], histSize=[8, 8, 8], ranges=[0, 256, 0, 256, 0, 256]):
    # ----------
    hist1 = cv2.calcHist([img1], channels=channels, mask=None, histSize=histSize, ranges=ranges)
    hist2 = cv2.calcHist([img2], channels=channels, mask=None, histSize=histSize, ranges=ranges)
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    # hist1 = hist1.flatten()
    # hist2 = hist2.flatten()
    # ----------
    results = {}
    results['Intersection'] = np.round(cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT), 4)
    results['Chi-Sqaured'] = np.round(chi2_distance(hist1, hist2), 4)
    results['Chebyshev'] = np.round(dist.chebyshev(hist1, hist2), 4)
    return results


def calc_color_iou(img1, img2, channels=[0, 1, 2], histSize=[8, 8, 8], ranges=[0, 256, 0, 256, 0, 256]):
    # ----------
    hist1 = cv2.calcHist([img1], channels=channels, mask=None, histSize=histSize, ranges=ranges)
    hist2 = cv2.calcHist([img2], channels=channels, mask=None, histSize=histSize, ranges=ranges)
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()
    # ----------
    sm = 0
    for i in range(len(hist1)):
        sm += min(hist1[i], hist2[i])
    iou1 = sm / sum(hist1)
    iou2 = sm / sum(hist2)
    return 2 * iou1 * iou2 / (iou1 + iou2)


# --------------------------------------------------------------------------------------------------
# edge
# --------------------------------------------------------------------------------------------------

def canny_edge(img_gray, ksize=(3,3), sigma=0.33):
    img_blur = cv2.GaussianBlur(img_gray, ksize, 0)
    med_val = np.median(img_blur)
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))
    return cv2.Canny(img_blur, threshold1=min_val, threshold2=max_val)


def calc_edge_iou(img1, img2, step_w=20, step_h=20):
    w, h = img1.shape
    count_w = w // step_w
    count_h = h // step_h
    edge_count1, edge_count2 = [], []
    for i in range(count_h + 1):
        hmin = step_h * i
        hmax = hmin + step_h
        if i == count_h:
            hmax = h
        for j in range(count_w + 1):
            wmin = step_w * j
            wmax = wmin + step_w
            if j == count_h:
                wmax = w
            edge_count1.append(len(np.where(img1[hmin:hmax, wmin:wmax] == 255)[0]))
            edge_count2.append(len(np.where(img2[hmin:hmax, wmin:wmax] == 255)[0]))
    sm = 0
    for i in range(len(edge_count1)):
        sm += min(edge_count1[i], edge_count2[i])
    iou1 = sm / sum(edge_count1)
    iou2 = sm / sum(edge_count2)
    return 2 * iou1 * iou2 / (iou1 + iou2)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# SSIM and CW-SSIM
# --------------------------------------------------------------------------------------------------

def comp_stats(img_path, img_path2, transform, cw_width=30, hist_bin_size=32):
    # ----------
    img1 = cv2.imread(img_path)
    img2 = cv2.imread(img_path2)
    h, w, _ = img1.shape
    img2 = cv2.resize(img2, (w, h))
    # ----------
    img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img_rgb)
    trans_img = transformed["image"]
    img2 = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
    # PIL.Image.fromarray(trans_img).show()
    # ----------
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img_lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    # to corresponds to skimage rgb2lab
    img_lab_f1 = cv2.cvtColor(img1.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    img_lab_f2 = cv2.cvtColor(img2.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    img_f1 = img1.astype(np.float32) / 255
    img_f2 = img2.astype(np.float32) / 255
    img_edge1 = canny_edge(img_gray1, ksize=(3, 3), sigma=0.33)
    img_edge2 = canny_edge(img_gray2, ksize=(3, 3), sigma=0.33)
    # PIL.Image.fromarray(img_edge1).show()
    # PIL.Image.fromarray(img_edge2).show()
    # ----------
    # hist_res_bgr = calc_hist_dist(img1, img2, channels=[0, 1, 2], histSize=[hist_bin_size, hist_bin_size, hist_bin_size], ranges=[0, 256, 0, 256, 0, 256])
    # hist_res_ab = calc_hist_dist(img_lab1, img_lab2, channels=[1, 2], histSize=[hist_bin_size, hist_bin_size], ranges=[0, 256, 0, 256])
    bgr_iou = calc_color_iou(img1, img2, channels=[0, 1, 2], histSize=[hist_bin_size, hist_bin_size, hist_bin_size], ranges=[0, 256, 0, 256, 0, 256])
    ab_iou = calc_color_iou(img_lab1, img_lab2, channels=[1, 2], histSize=[hist_bin_size, hist_bin_size], ranges=[0, 256, 0, 256])
    edge_iou = calc_edge_iou(img_edge1, img_edge2, step_w=max(int(w/20), 20), step_h=max(int(h/20), 20))
    # ----------
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    # ----------
    l = 255
    k_1 = 0.01
    k_2 = 0.03
    k = 0.01
    # ----
    mode = 'mean'
    # mode = 'max'
    ssim_val = ssim_val_func(img_gray1, img_gray2, gaussian_kernel_1d, l=l, k_1=k_1, k_2=k_2, k=k)
    cwssim_val = cwssim_val_func(img_gray1, img_gray2, k=k, cw_width=cw_width)
    ssim_mch_val = ssim_mch_val_func(img1, img2, gaussian_kernel_1d, l=l, k_1=k_1, k_2=k_2, k=k, mode=mode)
    cwssim_mch_val = cwssim_mch_val_func(img1, img2, k=k, cw_width=cw_width, mode=mode)
    # ----------
    ssim_val2, ssim_img2 = structural_similarity(img_gray1, img_gray2,
                                               data_range=255, full=True)
    ssim_val3, ssim_img3 = structural_similarity(img1, img2,
                                                 win_size=7, data_range=255,
                                                 channel_axis=-1, full=True)
    ssim_val4, ssim_img4 = structural_similarity(img1, img2,
                                                 channel_axis=-1, full=True,
                                                 gaussian_weights=True, use_sample_covariance=True)
    # ----------
    # KL = 1
    KL = 100
    color_diff = np.average(deltaE_ciede2000(img_lab_f1, img_lab_f2, kL=KL, kC=1, kH=1, channel_axis=-1))
    # ----------
    ssim_img_recov3 = np.clip(ssim_img3 * 255, 0, 255).astype('uint8')
    ssim_img_recov4 = np.clip(ssim_img4 * 255, 0, 255).astype('uint8')
    img_to_show = np.hstack([img1, img2, ssim_img_recov3, ssim_img_recov4])
    PIL.Image.fromarray(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)).show()
    PIL.Image.fromarray(np.hstack([img_edge1, img_edge2])).show()
    # ----------
    print('---------------------------------------------------------')
    print(f'img1: {os.path.basename(img_path)}')
    print(f'img2: {os.path.basename(img_path2)}')
    print('---------------------------------------------------------')
    print(f'SSIM value (gray):               {ssim_val: .4f}')
    print(f'CW-SSIM value (gray):            {cwssim_val: .4f}')
    print(f'DeltaE CIEDE2000 (float):        {color_diff: .4f}')
    print('---------------------------------------------------------')
    # print(f'Color Hist Dist (BGR):           {hist_res_bgr}')
    # print(f'Color Hist Dist (LAB):           {hist_res_ab}')
    print(f'Color IoU (BGR):                 {bgr_iou: .4f}')
    print(f'Color IoU (AB):                  {ab_iou: .4f}')
    print(f'Edge IoU:                        {edge_iou: .4f}')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print(f'SSIM value (mch):                {ssim_mch_val: .4f}  mode: {mode}')
    print('---------------------------------------------------------')
    print(f'SSIM value (gray):               {ssim_val2: .4f}')
    print(f'SSIM value (mch):                {ssim_val3: .4f}')
    print(f'SSIM value (mch + use_sample):   {ssim_val4: .4f}')
    print('---------------------------------------------------------')
    print(f'CW-SSIM value (mch):             {cwssim_mch_val: .4f}  mode: {mode}')


# --------------------------------------------------------------------------------------------------
# set 2 image paths
# --------------------------------------------------------------------------------------------------

# img_dir = '/home/kswada/kw/image_processing/00_sample_images/ssim'

# img_fname1 = 'test2-1.png'
# img_fname2 = 'test2-2.png'

# img_fname1 = 'test3-orig.jpg'
# img_fname2 = 'test3-cro.jpg'
# img_fname2 = 'test3-lig.jpg'
# img_fname2 = 'test3-rot.jpg'

# img_fname1 = 'camera.png'
# # img_fname2 = 'camera_noise.png'  # SSIM is 0.15 (by skimage.metrics.structural_similarity)
# img_fname2 = 'camera_const.png'  # SSIM is 0.85 (by skimage.metrics.structural_similarity)


# ----------
img_dir = '/home/kswada/kw/image_processing/00_sample_images/prize/brassband_trading_badge'
img_fname1 = 'brassband_trading_badge_01.jpg'
# img_fname2 = 'brassband_trading_badge_02.jpg'
# img_fname2 = 'brassband_trading_badge_03.jpg'
# # img_fname2 = 'brassband_trading_badge_04.jpg'
# # img_fname2 = 'brassband_trading_badge_05.jpg'
# img_fname2 = 'brassband_trading_badge_06.jpg'
img_fname2 = 'brassband_trading_badge_07.jpg'


# ----------
# img_dir = '/home/kswada/kw/image_processing/00_sample_images/skimage_data'
# img_fname1 = 'chelsea.png'
# img_fname2 = 'chelsea.png'
# img_fname1 = 'chessboard_RGB.png'
# img_fname2 = 'chessboard_RGB.png'
# img_fname1 = 'chessboard_GRAY.png'
# img_fname2 = 'chessboard_GRAY.png'
# img_fname1 = 'ihc.png'
# img_fname2 = 'ihc.png'
# img_fname1 = 'motorcycle_left.png'
# # # img_fname2 = 'motorcycle_left.png'
# img_fname2 = 'motorcycle_right.png'
# img_fname1 = 'page.png'
# img_fname2 = 'page.png'
# img_fname1 = 'grass.png'
# img_fname2 = 'grass.png'
# img_fname1 = 'gravel.png'
# img_fname2 = 'gravel.png'
# img_fname1 = 'logo.png'
# img_fname2 = 'logo.png'


# ----------
# img_dir = '/home/kswada/kw/image_processing/00_sample_images/cushion/pacman'
# # img_fname1 = 'cushion_galaxy_donuts_ghostbig_pacman_blue.jpg'
# img_fname2 = 'cushion_galaxy_donuts_ghostbig_pacman_orange.jpg'
# img_fname1 = 'cushion_galaxy_donuts_ghostbig_pacman_pink.jpg'
# # # img_fname2 = 'cushion_pacman_pink.jpg'
# # img_fname2 = 'cushion_pacman_ura_blue.jpg'


# ----------
# img_dir = '/home/kswada/kw/image_processing/00_sample_images/cushion'
# # # img_fname1 = 'cushion_01.jpg'
# # # # img_fname2 = 'cushion_03.jpg'
# img_fname1 = 'cushion_blown.jpg'
# # img_fname2 = 'cushion_07.jpg'
# img_fname2 = 'cushion_green_gray.jpg'


# ----------
img_path = os.path.join(img_dir, img_fname1)
img_path2 = os.path.join(img_dir, img_fname2)

h, w, _ = cv2.imread(img_path).shape

# ----------
transform = A.Compose([
    # A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(brightness_limit=(-0.2*2, 0.2*2), contrast_limit=(-0.2*2, 0.2*2), p=1.0),
    # A.Rotate((45, 45), p=1.0),
    # A.HueSaturationValue((-20*4, 20*4), (-30*4, 30*4), (-20*4, 20*4), p=1.0),
    # A.ChannelShuffle(p=1.0),
    # A.InvertImg(p=1.0),
    # A.RandomSizedCrop((200, 200), height=h, width=w, p=1.0)
])

# ----
# real zoomed image require larger cw_width (if smaller, sutle difference is captured and CW-SSIM value will be lower)
# cw_width = 30

# if design image smaller cw_width to capture global difference
# smaller to detect fine structural change
# cw_width = int(min(img1.shape[0], img1.shape[1]) / 20)
cw_width = 10


# ----
hist_bin_size = 64

comp_stats(img_path, img_path2, transform=transform, cw_width=cw_width, hist_bin_size=hist_bin_size)



####################################################################################################
# --------------------------------------------------------------------------------------------------
# SSIM and CW-SSIM:  step by step
# --------------------------------------------------------------------------------------------------
####################################################################################################

# --------------------------------------------------------------------------------------------------
# load base image
# --------------------------------------------------------------------------------------------------

img_dir = '/home/kswada/kw/image_processing/00_sample_images/ssim'

# img_fname_base = 'test2-1.png'
# img_fname_target = 'test2-2.png'

# img_fname_base = 'test3-orig.jpg'
# # img_fname_target = 'test3-cro.jpg'
# # img_fname_target = 'test3-lig.jpg'
# img_fname_target = 'test3-rot.jpg'

img_fname_base = 'camera.png'
# img_fname_target = 'camera_noise.png'
img_fname_target = 'camera_const.png'


img_path = os.path.join(img_dir, img_fname_base)
img_target_path = os.path.join(img_dir, img_fname_target)


# ----------
img = PIL.Image.open(img_path)
img_tgt = PIL.Image.open(img_target_path)

if img_tgt.size != img.size:
    img_tgt = img_tgt.resize(img.size, PIL.Image.ANTIALIAS)

img_size = img.size
img_size_tgt = img_tgt.size

# check if same size !!!
print(img_size)
print(img_size_tgt)


# ----------
imbands = img.getbands()
imbands_tgt = img_tgt.getbands()

print(imbands)
print(imbands_tgt)


# --------------------------------------------------------------------------------------------------
# gaussian kernel 1d
# --------------------------------------------------------------------------------------------------

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

print(gaussian_kernel_1d)


# --------------------------------------------------------------------------------------------------
# preprocess image from SSIM
# --------------------------------------------------------------------------------------------------

# np.array of grayscale and alpha image
img_gray = np.asarray(PIL.ImageOps.grayscale(img)).astype(float)
img_gray_tgt = np.asarray(PIL.ImageOps.grayscale(img_tgt)).astype(float)

print(img_gray)
print(img_gray.shape)

# ----------
img_alpha = None
img_alpha_tgt = None

if 'A' in imbands:
    img_alpha = np.asarray(img.split()[-1]).astype(float)

if 'A' in imbands_tgt:
    img_alpha_tgt = np.asarray(img_tgt.split()[-1]).astype(float)


if img_alpha is not None:
    img_gray[img_alpha == 255] = 0

if img_alpha_tgt is not None:
    img_gray_tgt[img_alpha_tgt == 255] = 0


# ----------
# Squared grayscale
img_gray_squared = img_gray ** 2
img_gray_squared_tgt = img_gray_tgt ** 2


# ----------
# Convolve grayscale image with gaussian
img_gray_mu = convolve_gaussian_2d(img_gray, gaussian_kernel_1d)
img_gray_mu_tgt = convolve_gaussian_2d(img_gray_tgt, gaussian_kernel_1d)


# Squared mu
img_gray_mu_squared = img_gray_mu ** 2
img_gray_mu_squared_tgt = img_gray_mu_tgt ** 2


# Convolve squared grayscale with gaussian
img_gray_sigma_squared = convolve_gaussian_2d(img_gray_squared, gaussian_kernel_1d)
img_gray_sigma_squared_tgt = convolve_gaussian_2d(img_gray_squared_tgt, gaussian_kernel_1d)


# Subtract squared mu
img_gray_sigma_squared -= img_gray_mu_squared
img_gray_sigma_squared_tgt -= img_gray_mu_squared_tgt


# --------------------------------------------------------------------------------------------------
# set params
# --------------------------------------------------------------------------------------------------

l = 255
k_1 = 0.01
k_2 = 0.03
k = 0.01

c_1 = (k_1 * l) ** 2
c_2 = (k_2 * l) ** 2

print(f'c_1: {c_1}')
print(f'c_2: {c_2}')


# --------------------------------------------------------------------------------------------------
# compute SSIM value
# --------------------------------------------------------------------------------------------------

img_mat_12 = img_gray * img_gray_tgt

img_mat_sigma_12 = convolve_gaussian_2d(img_mat_12, gaussian_kernel_1d)

img_mat_mu_12 = img_gray_mu * img_gray_mu_tgt

img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12


# ----------
# Numerator of SSIM
num_ssim = ((2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2))

# Denominator of SSIM
den_ssim = (
        (img_gray_mu_squared + img_gray_mu_squared_tgt + c_1) *
        (img_gray_sigma_squared + img_gray_sigma_squared_tgt + c_2))

ssim_map = num_ssim / den_ssim


# ----------
ssim_val = np.average(ssim_map)

print(f'SSIM value: {ssim_val: .4f}')


# --------------------------------------------------------------------------------------------------
# compute CW-SSIM value
# --------------------------------------------------------------------------------------------------

# Define a width for the wavelet convolution
cw_width = 30
widths = np.arange(1, cw_width + 1)


# Use the image data as arrays
# slightly different (img_tgt is already resized by Image.ANTIALIAS)
# sig1 = np.asarray(PIL.ImageOps.grayscale(img).getdata())
# sig2 = np.asarray(PIL.ImageOps.grayscale(img_tgt).getdata())

img1 = cv2.imread(img_path)
img2 = cv2.imread(img_target_path)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
sig1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).reshape(-1)
sig2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).reshape(-1)

print(sig1)
print(sig2)
print(len(sig1))
print(len(sig2))


# ----------
# Convolution
cwtmatr1 = signal.cwt(sig1, signal.ricker, widths)
cwtmatr2 = signal.cwt(sig2, signal.ricker, widths)


# ----------
# Compute the first term
c1c2 = np.multiply(abs(cwtmatr1), abs(cwtmatr2))
c1_2 = np.square(abs(cwtmatr1))
c2_2 = np.square(abs(cwtmatr2))
num_ssim_1 = 2 * np.sum(c1c2, axis=0) + k
den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + k


# Compute the second term
c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + k
den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + k


# ----------
# Construct the result
ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)

# Average the per pixel results
cwssim_val = np.average(ssim_map)


print(f'SSIM value:    {ssim_val: .4f}')
print(f'CW-SSIM value: {cwssim_val: .4f}')


