from __future__ import division, print_function

import numpy as np
from scipy import signal

from sigflux.wave import cwtlets

def abthresh(x, thresh=1.):
    """
    Thresholding function, absolute value of signal less than 'thresh' is set to 0
    :param x:
    :param thresh:
    :return:
    """
    idx = np.abs(x)[:,:] < thresh
    x[idx] = 0
    return x


def lazy_icwt(cw, widths):
    scaling = np.array(widths.reshape(-1,1))
    scaling = 1/ scaling # works with non-rescaling wavelets, e.g. amplitude 1
    cw2 = scaling * cw
    combined = np.mean(cw2, axis=0)
    return combined

def cwtfiltbank(cw, widths, amin, amax):
    idxmin = (np.abs(widths-amin)).argmin()
    idxmax = (np.abs(widths-amax)).argmin() + 1
    return(cw[idxmin:idxmax], widths[idxmin:idxmax])

def wconv(sig, wavelet, width):
    return signal.cwt(sig, wavelet, [width])[0]

def acwt(sig, widths):
    """ Reconstruct an analytic signal using CWT"""
    magical_constant = 16.854880972 # this doesn't actually scale properly
    # Somehow, I'm still seing dependence on the number of widths.
    cwt_r = signal.cwt(sig, signal.ricker, widths)
    cwt_i = signal.cwt(sig, cwtlets.ricker_i, widths)
    a_sig = lazy_icwt(cwt_r, widths) + 1j * lazy_icwt(cwt_i, widths)
    return a_sig #* magical_constant

def wavediff1d(sig, width=10, padout_ratio=.1):
    """
    Calculate the approximate "derivative" of a signal using a Haar wavelet. 'width' specifies the amount of smoothing
    :param sig: Signal to take derivative of
    :param width: Haar wavelet width parameter. See scipy.signal.cwt
    :param padout_ratio: Amount to pad the signal in order to reduce edge effects
    :return:
    """
    pad_width = int(padout_ratio*len(sig))
    padded_sig = np.pad(sig, pad_width, 'edge')
    cw = signal.cwt(padded_sig, cwtlets.haar, (width,)).T
    return cw[pad_width:-pad_width]

def conv(img, kern, renorm=False):
    ift = signal.fftconvolve(img, kern, 'same')
    if renorm:
        ift /= np.amax(ift)
    return ift

def ricker2d(x, y, f=np.pi, n=0.5):
    r = (x**2 + y**2)**n
    return (1.0 - 2.0*(np.pi**2)*(f**2)*(r**2)) * np.exp(-(np.pi**2)*(f**2)*(r**2))
