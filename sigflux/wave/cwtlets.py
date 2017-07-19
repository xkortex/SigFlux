from __future__ import division, print_function

import numpy as np
from scipy import signal


"""
Wavelets for use with the scipy signal.cwt
"""


def rescale(sig):
    """
    Returns a scaled version of the supplied wavelet function, scaled according to window size
    :param sig:
    :return:
    """
    xyz = (1 / np.pi) ** 0.125 # no idea why this works for the ricker wavelet
    return lambda n, a: xyz * sig(n, a) / a**0.5


def haar(n, a):
    a = int(a)
    n = int(n)
    y = np.zeros(n, dtype=float)
    m = n // 2
    y[m - 2 * a:m] = 1
    y[m: m + 2 * a] = -1
    return y


def haaro(n, a):
    """Odd harr wavelet"""
    a = int(a)
    n = int(n)
    m = n // 2
    y = np.zeros(n, dtype=float)
    y[m - a:m + a] = 1
    y[m - 2 * a: m - a] = -1
    y[m + a: m + 2 * a] = -1
    return y


def rect(n, a):
    """Rectangular pulse"""
    a = int(a)
    n = int(n)
    m = n // 2
    y = np.zeros(n, dtype=float)
    y[m - a:m + a] = 1
    y[m - 2 * a: m - a] = 0
    y[m + a: m + 2 * a] = 0
    return y


def ricker_i(points, a):
    return np.imag(signal.hilbert(signal.ricker(points, a)))


def s_rect(n, a, d=0.5):
    """Rectangular pulse with duty cycle"""
    a = int(a)
    n = int(n)
    m = n // 2
    r = 1/d
    y = np.zeros(n, dtype=float)
    y[m - a:m + a] = 1
    y[m - 2 * a: m - a] = 0
    y[m + a: m + 2 * a] = 0
    return y


def s_morlet(n, a, z=2.):
    """
    Morlet with variable shape
    :param n:
    :param a:
    :param z: Shaping parameter, default 2.0. Hat2: z=1/sqrt(e)
    :return:
    """
    n = float(n)
    a = float(a)
    t = np.arange(-n / 2, n / 2) / n
    fc = n / (.5 * np.pi * a) # fc is critical frequency
    b = n ** 2 / a ** 2 / np.pi
    sig = np.exp(-b * t ** 2) * np.exp(1j * z * np.pi * fc * t)
    return sig


def morlet(n, a):
    return s_morlet(n, a)


def gausslet(n, a):
    return s_morlet(n, a, 0.)


def igausslet(n, a):
    # return signal.hilbert(CWTlets.s_morlet(n, a, 0.))
    return np.imag(signal.hilbert(np.real(gausslet(n, a)))) # barf.


def bouncelet(n, a, z=2.):
    n = float(n)
    a = float(a)
    t = np.arange(-n / 2, n / 2) / n
    fc = n / (.5 * np.pi * a)  # fc is critical frequency
    b = n ** 2 / a ** 2 / np.pi
    sig = np.exp(-b * t ** 2) * np.exp(1j * z * np.pi * fc * t)
    sig = np.abs(np.real(sig))
    return sig


def morlet_hat(n, a):
    coef = 1/np.sqrt(np.e)
    return s_morlet(n, a, coef)

def hatlet(n, a):
    """
    Complex Mexican Hat (Ricker) wavelet
    :param n: Number of points
    :type n: int
    :param a: Width parameter of the wavelet
    :type a: float
    :return:
    """
    return signal.hilbert(signal.ricker(n, a))


def gen_morlet(z):
    return lambda n, a: s_morlet(n, a, z)


def sinclet(n, a):
    n = float(n)
    a = float(a)
    fc = n / (a) # fc is critical frequency
    t = np.arange(-n / 2, n / 2) / n
    fct = fc * t
    real = np.sin(fct) / (fct)
    imag = 1j * (-np.cos(fct) / (fct) + np.sin(fct) / fct ** 2)
    return real + imag


def sqsinc(n, a):
    n = float(n)
    a = float(a)
    fc = n / (a) # fc is critical frequency
    t = np.arange(-n / 2, n / 2) / n
    fct = fc * t
    real = (np.sin(fct) / (fct)) **2
    return real