import numpy as np
from numpy.fft import fftpack
from scipy import signal
from scipy.signal import convolve, fftconvolve


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    startind -= 1 # this is a hack to get bogo_fftconv to behave the same.
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]



def filterbank_cwt(data, wavelet, widths):
    N = len(data)
    wavelet_data = wavelet(10, 2)
    output = np.zeros([len(widths), len(data)], dtype=wavelet_data.dtype)
    wavebank = []
    for width in widths:
        wave = wavelet(min(10 * width, N), width).reshape(1, -1)
        p = (N - wave.shape[1]) // 2
        q = N - wave.shape[1] - p
        wave = np.pad(wave, [(0, 0), (p, q)], 'constant')
        wavebank.append(wave)
    # print(p, wave.shape)

    wavebank = np.concatenate(wavebank, axis=0)
    #     print(wavebank.shape)

    #     for ind, width in enumerate(widths):
    #         wavelet_data = wavelet(min(10 * width, len(data)), width)
    #         output[ind, :] = fftconvolve(data, wavelet_data,
    #                                   mode='same')
    return wavebank

def cwt(data, wavelet, widths):
    """
    Continuous wavelet transform. Just like SciPy, but allows complex input/output

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).

    Notes
    -----
    ::

        length = min(10 * width[ii], len(data))
        cwt[ii,:] = signal.convolve(data, wavelet(length,
                                    width[ii]), mode='same')


    """
    wavelet_data = wavelet(10, 2)
    output = np.zeros([len(widths), len(data)], dtype=wavelet_data.dtype)
    for ind, width in enumerate(widths):
        wavelet_data = wavelet(min(10 * width, len(data)), width)
        output[ind, :] = fftconvolve(data, wavelet_data,
                                  mode='same')
    return output


def complex_cwt(data, wavelet, widths):
    """Faster Continuous Wavelet Transform with full support of complex wavelets
    """
    assert data.ndim == 1, "Only supports 1D input currently"
    siglen = data.shape[0]  # same as widths, since we built it. Actually we need to pad the data first
    in1 = data  # np.pad(data, [(siglen//2, siglen//2)], 'constant')
    in2 = filterbank_cwt(data, wavelet, widths)
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    shape = [data.shape[0] * 2]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    #     print(in1.shape, in2.shape)
    #     print('Shape: ', shape, 'Fslice', fslice)

    fftn = fftpack.fftn
    ifftn = fftpack.ifftn

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [sp.fftpack.helper.next_fast_len(int(d)) for d in shape]
    #     print('in1', in1.shape, 'fshape', fshape)

    sp1 = fftn(in1, [shape[0]])
    sp2 = fftn(in2, [shape[0]], axes=[1])
    ret = ifftn(sp1 * sp2, axes=[1])  # [fslice].copy()


    return _centered(ret, s1)




