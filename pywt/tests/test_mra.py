#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

from itertools import product
import numpy as np
from numpy.testing import assert_allclose
import pytest

import pywt

# Check that float32 and complex64 are preserved.  Other real types get
# converted to float64.
dtypes_in = [np.int8, np.float16, np.float32, np.float64, np.complex64,
             np.complex128]
dtypes_out = [np.float64, np.float32, np.float32, np.float64, np.complex64,
              np.complex128]

# tolerances used in accuracy comparisons
tol_single = 1e-6
tol_double = 1e-13


@pytest.mark.parametrize(
    "wav, mode, shape, transform",
    product(
        ["sym2", ],  # "bior1.1"],
        ["periodization"],
        [(128,), (128, 64), (24,)],
        ["swt", "dwt"],
    ),
)
def test_mra_imra_roundtrip(wav, mode, shape, transform):
    atol = rtol = 1e-7
    rstate = np.random.RandomState(1234)
    x = rstate.standard_normal(shape)
    for axis in range(x.ndim):
        if transform == 'swt':
            if axis != x.ndim - 1:
                continue  # only last axis supported for swt
            if x.ndim > 1:
                continue  # only 1D inputs supported by swt
            if not pywt.Wavelet(wav).orthogonal:
                continue  # will get a warning for bi-orthgonal case
        w = pywt.mra(x, wav, level=2, axis=axis, mode=mode,
                     transform=transform)
        xr = pywt.imra(w)
        assert_allclose(x, xr, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "wav, mode, shape, transform",
    product(
        ["sym2", ],  # "bior1.1"],
        ["periodization"],
        [(128,), (12, 16)],
        ["swtn", "dwtn"],
    ),
)
def test_mran_imran_roundtrip(wav, mode, shape, transform):
    atol = rtol = 1e-7

    rstate = np.random.RandomState(1234)
    x = rstate.standard_normal(shape)
    w = pywt.mran(x, wav, level=2, mode=mode, transform=transform)
    xr = pywt.imran(w)
    assert_allclose(x, xr, atol=atol, rtol=rtol)
    for axis in range(x.ndim):
        w = pywt.mran(x, wav, level=2, axes=(axis,), mode=mode,
                      transform=transform)
        xr = pywt.imran(w)
        assert_allclose(x, xr, atol=atol, rtol=rtol)
