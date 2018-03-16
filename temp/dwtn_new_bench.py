from __future__ import division, print_function, absolute_import

from itertools import product

import numpy as np

from pywt._c99_config import _have_c99_complex
from pywt._extensions._dwt import dwt_axis, idwt_axis
from pywt._utils import _wavelets_per_axis, _modes_per_axis


def dwtn_old(data, wavelet, mode='symmetric', axes=None):
    """
    Single-level n-dimensional Discrete Wavelet Transform.

    Parameters
    ----------
    data : array_like
        n-dimensional array with input data.
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or tuple of string, optional
        Signal extension mode used in the decomposition,
        see Modes (default: 'symmetric').  This can also be a tuple of modes
        specifying the mode to use on each axis in ``axes``.
    axes : sequence of ints, optional
        Axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes. A value of ``None`` (the
        default) selects all axes.

        Axes may be repeated, but information about the original size may be
        lost if it is not divisible by ``2 ** nrepeats``. The reconstruction
        will be larger, with additional values derived according to the
        ``mode`` parameter. ``pywt.wavedecn`` should be used for multilevel
        decomposition.

    Returns
    -------
    coeffs : dict
        Results are arranged in a dictionary, where key specifies
        the transform type on each dimension and value is a n-dimensional
        coefficients array.

        For example, for a 2D case the result will look something like this::

            {'aa': <coeffs>  # A(LL) - approx. on 1st dim, approx. on 2nd dim
             'ad': <coeffs>  # V(LH) - approx. on 1st dim, det. on 2nd dim
             'da': <coeffs>  # H(HL) - det. on 1st dim, approx. on 2nd dim
             'dd': <coeffs>  # D(HH) - det. on 1st dim, det. on 2nd dim
            }

        For user-specified ``axes``, the order of the characters in the
        dictionary keys map to the specified ``axes``.

    """
    data = np.asarray(data)
    if not _have_c99_complex and np.iscomplexobj(data):
        real = dwtn(data.real, wavelet, mode, axes)
        imag = dwtn(data.imag, wavelet, mode, axes)
        return dict((k, real[k] + 1j * imag[k]) for k in real.keys())

    if data.dtype == np.dtype('object'):
        raise TypeError("Input must be a numeric array-like")
    if data.ndim < 1:
        raise ValueError("Input data must be at least 1D")

    if axes is None:
        axes = range(data.ndim)
    axes = [a + data.ndim if a < 0 else a for a in axes]

    modes = _modes_per_axis(mode, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)

    coeffs = [('', data)]
    for axis, wav, mode in zip(axes, wavelets, modes):
        new_coeffs = []
        for subband, x in coeffs:
            cA, cD = dwt_axis(x, wav, mode, axis)
            new_coeffs.extend([(subband + 'a', cA),
                               (subband + 'd', cD)])
        coeffs = new_coeffs
    return dict(coeffs)


def dwtn(data, wavelet, mode='symmetric', axes=None, optimize=True):
    """
    Single-level n-dimensional Discrete Wavelet Transform.

    Parameters
    ----------
    data : array_like
        n-dimensional array with input data.
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or tuple of string, optional
        Signal extension mode used in the decomposition,
        see Modes (default: 'symmetric').  This can also be a tuple of modes
        specifying the mode to use on each axis in ``axes``.
    axes : sequence of ints, optional
        Axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes. A value of ``None`` (the
        default) selects all axes.

        Axes may be repeated, but information about the original size may be
        lost if it is not divisible by ``2 ** nrepeats``. The reconstruction
        will be larger, with additional values derived according to the
        ``mode`` parameter. ``pywt.wavedecn`` should be used for multilevel
        decomposition.
    optimize : bool, optional
        If True, the order in which the axes are filtered is optimized to
        reduce the amount of times subsets of `data` must be copied. If False,
        decomposition always proceeds in the order specified in `axes`.

    Returns
    -------
    coeffs : dict
        Results are arranged in a dictionary, where key specifies
        the transform type on each dimension and value is a n-dimensional
        coefficients array.

        For example, for a 2D case the result will look something like this::

            {'aa': <coeffs>  # A(LL) - approx. on 1st dim, approx. on 2nd dim
             'ad': <coeffs>  # V(LH) - approx. on 1st dim, det. on 2nd dim
             'da': <coeffs>  # H(HL) - det. on 1st dim, approx. on 2nd dim
             'dd': <coeffs>  # D(HH) - det. on 1st dim, det. on 2nd dim
            }

        For user-specified ``axes``, the order of the characters in the
        dictionary keys map to the specified ``axes``.

    """
    data = np.asarray(data)
    if not _have_c99_complex and np.iscomplexobj(data):
        real = dwtn(data.real, wavelet, mode, axes)
        imag = dwtn(data.imag, wavelet, mode, axes)
        return dict((k, real[k] + 1j * imag[k]) for k in real.keys())

    if data.dtype == np.dtype('object'):
        raise TypeError("Input must be a numeric array-like")
    if data.ndim < 1:
        raise ValueError("Input data must be at least 1D")

    if axes is None:
        axes = range(data.ndim)
    axes = [a + data.ndim if a < 0 else a for a in axes]

    modes = _modes_per_axis(mode, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)

    # Reorder (axes, modes, wavelets) for optimal computation time:
    #    Process any contiguous axis first (when the data is largest). This
    #    reduces the number of in-memory copies required later while filtering
    #    along the non-contiguous axes.
    axes_compute = axes
    if optimize and len(axes) > 1:
        if data.flags.c_contiguous:
            axes_compute = sorted(axes, reverse=True)
        elif data.flags.f_contiguous:
            axes_compute = sorted(axes, reverse=False)
        if axes != axes_compute:
            # reorder modes and wavelets to match axes_compute
            order = [axes.index(a) for a in axes_compute]
            modes = [modes[a] for a in order]
            wavelets = [wavelets[a] for a in order]

    coeffs = [('', data)]
    for axis, wav, mode in zip(axes_compute, wavelets, modes):
        new_coeffs = []
        for subband, x in coeffs:
            cA, cD = dwt_axis(x, wav, mode, axis)
            new_coeffs.extend([(subband + 'a', cA),
                               (subband + 'd', cD)])
        coeffs = new_coeffs

    if optimize and axes != axes_compute:
        # Make sure keys match axes rather than axes_compute
        def _reorder_key(k, order):
            return ''.join([k[n] for n in order])
        coeffs = {_reorder_key(k, order): v for k, v in new_coeffs}
    else:
        coeffs = dict(coeffs)

    return coeffs


from time import time
import pywt

nx = ny = 64
wav = pywt.Wavelet('db2')
z_sizes = [2**d for d in np.arange(4, 10.5, 0.2)]
# z_sizes = [2**d for d in range(3, 11)]
times_old = np.zeros(len(z_sizes))
times_new = np.zeros(len(z_sizes))
axes = None  # (0, 1, 2)
# axes = (2, 1, 0)
# mode = 'periodization'
mode = 'symmetric'
for i, nz in enumerate(z_sizes):
    # nreps = (2 * np.max(z_sizes)) // nz
    nz = int(nz)
    if nz % 2:
        nz += 1
    nreps = int(np.ceil(1 * np.max(z_sizes) / nz))
    print(nz, nreps)
    # if nz < 128:
    #     nreps = 64
    # elif nz < 1024:
    #     nreps = 16
    # else:
    #     nreps = 4
    x = np.random.randn(nx, ny, nz)

    tstart = time()
    for n in range(nreps):
        c = dwtn_old(x, wav, mode=mode, axes=axes)
    times_old[i] = (time() - tstart)/nreps

    tstart = time()
    for n in range(nreps):
        c = dwtn(x, wav, mode=mode, axes=axes, optimize=True)
    times_new[i] = (time() - tstart)/nreps

from matplotlib import pyplot as plt
plt.figure()
plt.plot(z_sizes, times_old, z_sizes, times_new)

plt.figure()
plt.plot(z_sizes, times_old/times_new)



wav = pywt.Wavelet('db2')
z_sizes = [2**d for d in np.arange(3, 8.5, 0.25)]
times_old = np.zeros(len(z_sizes))
times_new = np.zeros(len(z_sizes))
axes = (0, 1, 2)
# axes = (2, 1, 0)
for i, nz in enumerate(z_sizes):
    nz = int(nz)
    if nz % 2:
        nz += 1
    nreps = int(np.ceil(1 * np.max(z_sizes) / nz))
    print(nz, nreps)
    # if nz < 128:
    #     nreps = 64
    # elif nz < 1024:
    #     nreps = 16
    # else:
    #     nreps = 4
    x = np.random.randn(nz, nz, nz)
    tstart = time()
    for n in range(nreps):
        c = dwtn_old(x, wav, mode='periodization', axes=axes)
    times_old[i] = (time() - tstart)/nreps
    tstart = time()
    for n in range(nreps):
        c = dwtn(x, wav, mode='periodization', axes=axes)
    times_new[i] = (time() - tstart)/nreps


from matplotlib import pyplot as plt
plt.figure()
plt.plot(z_sizes, times_old, z_sizes, times_new)

plt.figure()
plt.plot(z_sizes, times_old/times_new)
