
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import PIL.Image

from scipy import signal

base_path = '/home/kswada/kw/image_processing'


####################################################################################################
# --------------------------------------------------------------------------------------------------
# BASICS:  skimage scipy signal cwt (continuous wavelet transform)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html
# --------------------------------------------------------------------------------------------------

len_t = 200
t = np.linspace(-1, 1, len_t, endpoint=False)
print(len(t))


sig = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)

cw_width = 30
widths = np.arange(1, cw_width+1)

# signal.ricker is Ricker wavelet (like Mexican Hat)
# the smaller cw_width, sharp (narrow) mother wavelet
cwtmatr = signal.cwt(sig, signal.ricker, widths)
# (cw_width, len_t)
print(cwtmatr.shape)

cwtmatr_yflip = np.flipud(cwtmatr)


# ----------
print('x(t) and Mother Wavelet')
plt.plot(sig)
plt.plot(signal.ricker(len_t, cw_width))
plt.show()


# ----------
plt.imshow(cwtmatr_yflip, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

