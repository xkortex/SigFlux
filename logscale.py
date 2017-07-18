import numpy as np
from scipy import fftpack, interpolate, signal
import clip


def freq_logscale(data, ndim=1024, fs=400, down=30, smoothing_cutoff=1, hard_cutoff=200, log_low_cut=-2.32,
                             prenormalize=True, useEnvelope=True):
    """
    This function returns a distorted version (x-axis log-transformed) of the fourier transform of the signal, related
    to the Mel scale (equal distance is equal temperment (pitch) not frequency)
    :param data: input data, [n,t] array-like
    :param ndim: dimension of output vector
    :param fs: input data sampling frequency
    :param smoothing_cutoff: 'Frequency' of smoothing the spectrum
    :param hard_cutoff: Chop off the frequency spectrum above this frequency
    :param log_low_cut: Sets how much to include very low frequency components
    :return:
    """
    if prenormalize:
        data = clip.norm_softclip(data)

    if useEnvelope:
        data = clip.envelope(data)

    # FFT and magnitude
    ftsig = fftpack.fft(data, axis=0)
    ftsig_a = np.abs(ftsig[:len(ftsig)*hard_cutoff//fs])
    # Smooth it with low pass and downsample. Low pass may not be necessary since resample does appropriate
    # pre-filtering
    ftsig_r = signal.resample_poly(ftsig_a, 1, down, axis=0)

    # Ok, now the weird bit. Take the existing x-domain and create an interpolation image of it
    t_rs = np.linspace(0.0001, hard_cutoff, len(ftsig_r))
    fn_ftsig_rs = interpolate.Akima1DInterpolator(t_rs, ftsig_r)
    # And now map an exponential domain, thereby creating a higher density of points around the main freq
    x_basis = np.linspace(log_low_cut, np.log2(hard_cutoff), ndim)
    log_ftsig = fn_ftsig_rs(np.power(2, x_basis))
    return log_ftsig