from math import floor, ceil

from ._extensions._pywt import (DiscreteContinuousWavelet, ContinuousWavelet,
                                Wavelet, _check_dtype)
from ._functions import integrate_wavelet, scale2frequency


__all__ = ["cwt"]


import numpy as np

try:
    from scipy.fftpack import next_fast_len
except ImportError:
    # Do provide a fallback so scipy is an optional requirement
    def next_fast_len(n):
        """Given a number of samples `n`, returns the next power of two
        following this number to take advantage of FFT speedup.
        This fallback is less efficient than `scipy.fftpack.next_fast_len`
        """
        return 2**ceil(np.log2(n))


def _reshape_nd(f, ndim, axis):
    # expand 1d f to ndim dimensions with size 1 along all but one axis.
    f_shape = [1] * ndim
    f_shape[axis] = f.size
    return f.reshape(tuple(f_shape))


def _slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    This function is copied from numpy's arraypad.py

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> _slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), (...,))
    """
    return (slice(None),) * axis + (sl,) + (...,)


def cwt(data, scales, wavelet, sampling_period=1., method='conv', axis=-1):
    """
    cwt(data, scales, wavelet)

    One dimensional Continuous Wavelet Transform.

    Parameters
    ----------
    data : array_like
        Input signal
    scales : array_like
        The wavelet scales to use. One can use
        ``f = scale2frequency(scale, wavelet)/sampling_period`` to determine
        what physical frequency, ``f``. Here, ``f`` is in hertz when the
        ``sampling_period`` is given in seconds.
    wavelet : Wavelet object or name
        Wavelet to use
    sampling_period : float
        Sampling period for the frequencies output (optional).
        The values computed for ``coefs`` are independent of the choice of
        ``sampling_period`` (i.e. ``scales`` is not scaled by the sampling
        period).
    method : {'conv', 'fft'}, optional
        The method used to compute the CWT. Can be any of:
            - ``conv`` uses ``numpy.convolve``.
            - ``fft`` uses frequency domain convolution via ``numpy.fft.fft``.
            - ``auto`` uses automatic selection based on an estimate of the
              computational complexity at each scale.
        The ``conv`` method complexity is ``O(len(scale) * len(data))``.
        The ``fft`` method is ``O(N * log2(N))`` with
        ``N = len(scale) + len(data) - 1``. It is well suited for large size
        signals but slightly slower than ``conv`` on small ones.
    axis: int, optional
        Axis over which to compute the DWT. If not given, the
        last axis is used.

    Returns
    -------
    coefs : array_like
        Continuous wavelet transform of the input signal for the given scales
        and wavelet
    frequencies : array_like
        If the unit of sampling period are seconds and given, than frequencies
        are in hertz. Otherwise, a sampling period of 1 is assumed.

    Notes
    -----
    Size of coefficients arrays depends on the length of the input array and
    the length of given scales.

    Examples
    --------
    >>> import pywt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(512)
    >>> y = np.sin(2*np.pi*x/32)
    >>> coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
    >>> plt.matshow(coef) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP
    ----------
    >>> import pywt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
    >>> widths = np.arange(1, 31)
    >>> cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
    >>> plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    ...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP
    """

    # accept array_like input; make a copy to ensure a contiguous array
    dt = _check_dtype(data)
    data = np.array(data, dtype=dt)
    if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if np.isscalar(scales):
        scales = np.array([scales])

    if axis < 0:
        axis = axis + data.ndim
    if not 0 <= axis < data.ndim:
        raise ValueError("Axis greater than data dimensions")

    dt_out = None  # TODO: fix in/out dtype consistency in a subsequent PR
    if wavelet.complex_cwt:
        dt_out = complex

    out = np.empty((scales.size, ) + data.shape, dtype=dt_out)
    precision = 10
    int_psi, x = integrate_wavelet(wavelet, precision=precision)

    if method == 'fft':
        size_scale0 = -1
        fft_data = None
    elif not method == 'conv':
        raise ValueError("method must be 'conv' or 'fft'")

    for i, scale in enumerate(scales):
        step = x[1] - x[0]
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
        j = j.astype(int)  # floor
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        int_psi_scale = int_psi[j][::-1]

        if method == 'conv':
            if data.ndim == 1:
                conv = np.convolve(data, int_psi_scale)
            else:
                conv = np.apply_along_axis(np.convolve,
                                           axis,
                                           data,
                                           int_psi_scale)
        else:
            # the padding is selected for
            # - optimal FFT complexity
            # - to be larger than the two signals length to avoid circular
            #   convolution
            size_scale = data.shape[axis] + int_psi_scale.size - 1
            size_scale = next_fast_len(size_scale)
            if size_scale != size_scale0:
                # the fft of data changes when padding size changes thus
                # it has to be recomputed
                fft_data = np.fft.fft(data, size_scale, axis=axis)
            size_scale0 = size_scale
            fft_wav = np.fft.fft(int_psi_scale, size_scale)
            if data.ndim > 1:
                fft_wav = _reshape_nd(fft_wav, data.ndim, axis=axis)
            conv = np.fft.ifft(fft_wav * fft_data, axis=axis)
            if not np.iscomplexobj(out):
                conv = conv.real
            sl = slice(data.shape[axis] + int_psi_scale.size - 1)
            conv = conv[_slice_at_axis(sl, axis)]
        coef = - np.sqrt(scale) * np.diff(conv, axis=axis)
        d = (coef.shape[axis] - data.shape[axis]) / 2.
        if d > 0:
            sl = slice(floor(d), -ceil(d))
            out[i, ...] = coef[_slice_at_axis(sl, axis)]
        elif d == 0.:
            out[i, ...] = coef
        else:
            raise ValueError(
                "Selected scale of {} too small.".format(scale))
    frequencies = scale2frequency(wavelet, scales, precision)
    if np.isscalar(frequencies):
        frequencies = np.array([frequencies])
    frequencies /= sampling_period
    return out, frequencies
