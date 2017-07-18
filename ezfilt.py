import numpy as np
from numpy import pi
from scipy import signal
import pandas as pd

def butter_lowpass_filter(data, cutoff, fs=1., order=1, axis=0, analog=False):
    # todo: add option to filtfilt or lfilter
    """
    Apply a digital Butterworth low-pass filter.
    :param data: array-like
    :param cutoff: Critical frequency, Hz
    :param fs: Sampling freqency, Hz
    :param order: Order of
    :param axis: ndarray axis, 0='long' axis, 1='row' axis
    :return:
    """
    nyquistFreqInRads = (2*pi*fs)/2
    Wn = 2*pi*cutoff / (nyquistFreqInRads)
    b, a = signal.butter(order, Wn, btype='low', analog=analog)
    y = signal.filtfilt(b, a, data, axis=axis)
    return y

def butterfilt(data, cutoff, fs=1., order=1, btype='low', ftype='filtfilt', axis=0, analog=False):
    """
    Apply a digital Butterworth low-pass filter.
    :param data: array-like
    :type data: np.ndarray pd.DataFrame
    :param cutoff: Critical frequency, Hz
    :param fs: Sampling freqency, Hz
    :param order: Order of
    :param axis: ndarray axis, 0='long' axis, 1='row' axis
    :return:
    """
    nyquistFreqInRads = (2*pi*fs)/2

    if isinstance(cutoff, (tuple, list, np.ndarray)):
        crit = 2*pi*np.array(cutoff) / nyquistFreqInRads
    else:
        crit = 2*pi*cutoff / nyquistFreqInRads
    b, a = signal.butter(order, crit, btype=btype, analog=analog)
    if ftype == 'filtfilt':
        y = signal.filtfilt(b, a, data, axis=axis)
    elif ftype == 'lfilt':
        y = signal.lfilter(b, a, data, axis=axis)
    else:
        raise ValueError('Invalid filter type specified: {}'.format(btype))

    # If data is dataframe, restore the original column and index info
    if isinstance(data, pd.DataFrame):
        y = pd.DataFrame(y, index=data.index, columns=data.columns)
    elif isinstance(data, pd.Series):
        y = pd.Series(y, index=data.index, name=data.name)

    return y

def filt(data, cutoff, fs=1., order=1, rp=10., rs=10., kind='butter', btype='low', ftype='filtfilt', axis=0, analog=False):
    """
    Apply a digital filter.
    :param data:
    :param cutoff:
    :param fs:
    :param order:
    :param rp:
    :param rs:
    :param kind:
    :param btype:
    :param ftype:
    :param axis:
    :param analog:
    :return:
    """
    nyquistFreqInRads = (2*pi*fs)/2
    crit = 2*pi*cutoff / nyquistFreqInRads
    if kind == 'butter':
        b, a = signal.butter(order, crit, btype=btype, analog=analog)
    elif kind == 'bessel':
        b, a = signal.bessel(order, Wn=crit, btype=btype, analog=analog)
    elif kind == 'cheby1':
        b, a = signal.cheby1(order, rp=rp, Wn=crit, btype=btype, analog=analog)
    elif kind == 'cheby2':
        b, a = signal.cheby2(order, rs=rs, Wn=crit, btype=btype, analog=analog)
    elif kind == 'ellip':
        b, a = signal.ellip(order, rp=rp, rs=rs, Wn=crit, btype=btype, analog=analog)
    else:
        raise ValueError('Invalid filter type specified: {}'.format(kind))

    if ftype == 'filtfilt':
        y = signal.filtfilt(b, a, data, axis=axis)
    elif ftype == 'lfilt':
        y = signal.lfilter(b, a, data, axis=axis)
    else:
        raise ValueError('Invalid filter type specified: {}'.format(btype))

    # If data is dataframe, restore the original column and index info
    if isinstance(data, pd.DataFrame):
        y = pd.DataFrame(y, index=data.index, columns=data.columns)
    elif isinstance(data, pd.Series):
        y = pd.Series(y, index=data.index, name=data.name)

    return y