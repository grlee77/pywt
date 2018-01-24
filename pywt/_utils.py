# Copyright (c) 2017 The PyWavelets Developers
#                    <https://github.com/PyWavelets/pywt>
# See COPYING for license details.
import numpy as np
import sys
from collections import Iterable

from ._extensions._pywt import (Wavelet, ContinuousWavelet,
                                DiscreteContinuousWavelet, Modes)


# define string_types as in six for Python 2/3 compatibility
if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,


__all__ = ['demo_function']


def _as_wavelet(wavelet):
    """Convert wavelet name to a Wavelet object"""
    if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if isinstance(wavelet, ContinuousWavelet):
        raise ValueError(
            "A ContinuousWavelet object was provided, but only discrete "
            "Wavelet objects are supported by this function.  A list of all "
            "supported discrete wavelets can be obtained by running:\n"
            "print(pywt.wavelist(kind='discrete'))")
    return wavelet


def _wavelets_per_axis(wavelet, axes):
    """Initialize Wavelets for each axis to be transformed.

    Parameters
    ----------
    wavelet : Wavelet or tuple of Wavelets
        If a single Wavelet is provided, it will used for all axes.  Otherwise
        one Wavelet per axis must be provided.
    axes : list
        The tuple of axes to be transformed.

    Returns
    -------
    wavelets : list of Wavelet objects
        A tuple of Wavelets equal in length to ``axes``.

    """
    axes = tuple(axes)
    if isinstance(wavelet, string_types + (Wavelet, )):
        # same wavelet on all axes
        wavelets = [_as_wavelet(wavelet), ] * len(axes)
    elif isinstance(wavelet, Iterable):
        # (potentially) unique wavelet per axis (e.g. for dual-tree DWT)
        if len(wavelet) == 1:
            wavelets = [_as_wavelet(wavelet[0]), ] * len(axes)
        else:
            if len(wavelet) != len(axes):
                raise ValueError((
                    "The number of wavelets must match the number of axes "
                    "to be transformed."))
            wavelets = [_as_wavelet(w) for w in wavelet]
    else:
        raise ValueError("wavelet must be a str, Wavelet or iterable")
    return wavelets


def _modes_per_axis(modes, axes):
    """Initialize mode for each axis to be transformed.

    Parameters
    ----------
    modes : str or tuple of strings
        If a single mode is provided, it will used for all axes.  Otherwise
        one mode per axis must be provided.
    axes : tuple
        The tuple of axes to be transformed.

    Returns
    -------
    modes : tuple of int
        A tuple of Modes equal in length to ``axes``.

    """
    axes = tuple(axes)
    if isinstance(modes, string_types + (int, )):
        # same wavelet on all axes
        modes = [Modes.from_object(modes), ] * len(axes)
    elif isinstance(modes, Iterable):
        if len(modes) == 1:
            modes = [Modes.from_object(modes[0]), ] * len(axes)
        else:
            # (potentially) unique wavelet per axis (e.g. for dual-tree DWT)
            if len(modes) != len(axes):
                raise ValueError(("The number of modes must match the number "
                                  "of axes to be transformed."))
        modes = [Modes.from_object(mode) for mode in modes]
    else:
        raise ValueError("modes must be a str, Mode enum or iterable")
    return modes


def demo_function(n, name='Bumps'):
    """Simple 1D wavelet test functions.

    This set of test functions were originally proposed in [1]_.

    Parameters
    ----------
    n : int
        The length of the test signal.
    name : {'Blocks', 'Bumps', 'HeaviSine', 'Doppler'}
        The type of test signal to generate (`name` is case-insensitive).

    Returns
    -------
    f : np.ndarray
        Array of length ``n`` corresponding to the specified test signal type.

    References
    ----------
    .. [1] D.L. Donoho and I.M. Johnstone.  Ideal spatial adaptation by
        wavelet shrinkage. Biometrika, vol. 81, pp. 425–455, 1994.
    """

    if n < 1 or (n % 1) != 0:
        raise ValueError("n must be an integer >= 1")
    # t = np.linspace(0, 1, n)
    t = np.arange(0, 1, 1/n)
    name = name.lower()
    if name == "blocks":
        t0s = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]
        hs = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]
        f = 0
        for (t0, h) in zip(t0s, hs):
            f += h * (1 + np.sign(t - t0)) / 2
    elif name == "bumps":
        t0s = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]
        hs = [4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2]
        ws = [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008,
              0.005]
        f = 0
        for (t0, h, w) in zip(t0s, hs, ws):
            f += h / (1 + np.abs((t - t0) / w))**4
    elif name == "heavisine":
        f = 4 * np.sin(4 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)
    elif name == "doppler":
        f = np.sqrt(t * (1 - t)) * np.sin(2 * np.pi * 1.05 / (t + 0.05))
    else:
        raise ValueError("unknown test function")
    return f
