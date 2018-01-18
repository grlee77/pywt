"""Cost functions for determining a best basis for wavelet packet transforms.

These functions have the general property of describing the concentration of
the signal energy into relatively few coefficients.

Notes
-----
When comparing the cost of a parent node to the summed costs of it's children
it is assumed that the energy is preserved (i.e.
``np.linalg.norm(coeffs) == np.linalg.norm(signal)``).  This is only strictly
true if the Wavelet is orthogonal and if mode == 'periodization'.  For other
modes, the extra boundary coefficients introduced will result in:
    np.linalg.norm(coeffs) >= np.linalg.norm(signal).

References
----------
.. [1] M. V. Wickerhauser. Adapted Wavelet Analysis from Theory to Software.
    Wellesley, MA: AK Peters, Ltd., 1994.

"""
import numpy as np


def cost_thresh(c, thresh):
    """Threshold-based cost function.

    Cost us determined by the number of coefficients with magnitude greater
    than `thresh`

    Parameters
    ----------
    c : np.ndarray
        Signal coefficients.
    thresh : float
        The threshold to use.
    """
    return np.sum(np.abs(c) > thresh)


def cost_l1(c):
    """Cost based on sum of the absolute value of the coefficients.

    Parameters
    ----------
    c : np.ndarray
        Signal coefficients.

    Returns
    -------
    cost : float
        The cost.
    """
    return np.sum(np.abs(c))


def cost_lp(c, p=1.5):
    """Cost based on the lp-norm of the values in c

    Parameters
    ----------
    c : np.ndarray
        Signal coefficients.
    p : float
        The exponent used in the norm (i.e. `ord` for `np.linalg.norm`).

    Returns
    -------
    cost : float
        The cost.
    """
    if p == 1:
        return cost_l1(c)
    c = np.atleast_1d(c)
    c = np.asarray(c, dtype=np.result_type(c.dtype, np.float32))
    return np.linalg.norm(c.ravel(order='K'), ord=p)


def cost_entropy(c):
    """Cost based on the Shannon-Weaver entropy of a sequence.

    This cost was proposed in [1]_.

    Parameters
    ----------
    c : np.ndarray
        Signal coefficients.

    Returns
    -------
    cost : float
        The cost.

    Notes
    -----
    This equation should be used on signals with l2-norm = 1.  In other words,
    the input to the wavelet packet transform should have an l2 norm of 1.

    References
    ----------
    .. [1] R.R. Coifman and M.V. Wickerhauser.  Entropy-Based Algorithms for
        Best Basis Selection.  IEEE Trans. Inf. Theory. Vol. 38, No. 2, 1992.
    .. [2] C.E. Shannon and W. Weaver.  The Mathematical Theory of
        Communication.  The University of Illinois Press, Urbana, 1964.

    """
    c = np.atleast_1d(c)
    c = np.asarray(c, dtype=np.result_type(c.dtype, np.float32))
    eps = np.finfo(c.dtype).eps
    if np.iscomplexobj(c):
        c2 = np.real(np.conj(c)*c)
    else:
        c2 = c * c
    neg_cost = np.sum(c2 * np.log(c2 + eps))
    if neg_cost == 0:
        return 0.  # avoid returning -0.0
    return -neg_cost


def cost_gauss_markov(c):
    """Cost based on Gauss-Markov information.

    Parameters
    ----------
    c : np.ndarray
        Signal coefficients.

    Returns
    -------
    cost : float
        The cost.

    References
    ----------
    .. [1] M. V. Wickerhauser. Adapted Wavelet Analysis: From Theory to
        Software. Wellesley, MA; AK Peters, Ltd., 1994.

    """
    c = np.atleast_1d(c)
    c = np.asarray(c, dtype=np.result_type(c.dtype, np.float32))
    if np.iscomplexobj(c):
        cost = np.real(np.conj(c)*c)
    else:
        cost = c * c
    np.log(cost, out=cost)
    return np.sum(cost)


def theoretical_dimension(c):
    """The "theoretical dimension" of a sequence.

    This quantity was defined in [1]_.

    Parameters
    ----------
    c : np.ndarray
        Signal coefficients.

    Returns
    -------
    cost : float
        The cost.

    References
    ----------
    .. [1] R.R. Coifman and M.V. Wickerhauser.  Entropy-Based Algorithms for
        Best Basis Selection.  IEEE Trans. Inf. Theory. Vol. 38, No. 2, 1992.
    """
    c = np.atleast_1d(c)
    c = np.asarray(c, dtype=np.result_type(c.dtype, np.float32))
    if np.iscomplexobj(c):
        c2 = np.real(np.conj(c)*c)
    else:
        c2 = c * c
    e = np.sum(c2)
    if e == 0:
        return 0.
    c2 /= e
    d = -np.sum(c2 * np.log(c2))
    return np.exp(d)


if False:
    # TODO: remove this demo
    import numpy as np
    import pywt
    from scipy.signal import chirp
    t = np.linspace(0, 10, 4096)
    r = chirp(t, f0=6, f1=0.1, t1=10, method='linear')
    # r = np.random.randn(4096)
    r /= np.linalg.norm(r)

    # single-level dwt
    c = pywt.dwt(r, 'db6', mode='periodization')
    print(cost_entropy(r))
    print(cost_entropy(c[0]) + cost_entropy(c[1]))
    print(cost_entropy(np.concatenate(c, axis=0)))
    c2 = pywt.dwt(r, 'bior4.4', mode='periodization')
    print(cost_entropy(r))
    print(cost_entropy(c2[0]) + cost_entropy(c2[1]))
    print(cost_entropy(np.concatenate(c2, axis=0)))

    # multi-level
    c = pywt.wavedec(r, 'db8', mode='periodization')
    c = pywt.coeffs_to_array(c)[0]
    print(cost_entropy(r))
    print(cost_entropy(c))

    c2 = pywt.wavedec(r, 'bior4.4', mode='periodization')
    c2 = pywt.coeffs_to_array(c2)[0]
    print(cost_entropy(r))
    print(cost_entropy(c2))
    # for bio-orthogonal case should be rescaling by the norm at this level
    print(cost_entropy(c2/np.linalg.norm(c2)))

    # multi-level non-periodization.  the extra coefficients introduced raise
    # the cost
    c3 = pywt.wavedec(r, 'bior4.4', mode='symmetric')
    c3 = pywt.coeffs_to_array(c3)[0]
    print(cost_entropy(r))
    print(cost_entropy(c3))
