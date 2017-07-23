import numpy as np
from sigflux.wave.custom_cwt import complex_cwt
from sigflux.wave.cwtlets import hatlet
from sklearn.cluster import KMeans

def pfa(data, widths):
    """
    Principle Frequency Analysis using Continuous Wavelet Transform
    :param data:
    :param widths:
    :return:
    """
    analytic_cw = complex_cwt(data, hatlet, widths)
    idx = np.argmax(np.abs(analytic_cw), axis=0)
    # components = list(set(idx))

    # return analytic_cw[components], components