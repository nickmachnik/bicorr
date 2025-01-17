"""
Computation of biserial correlations in python.
"""

import numpy as np
from scipy.stats import norm

def biserial_correlation(cont_data: np.array, bin_data: np.array, rm_nan=True) -> (float, float):
    """Compute biserial correlation and its standard error.

    Args:
        cont_data (array like): data of continuous variable
        bin_data (array like): data of binary variable
        rm_nan (bool): use only observations that are not nan in both arrays

    Returns:
        Tuple(float float): correlation, standard error
    """
    if rm_nan:
        keep = ~np.isnan(cont_data) & ~np.isnan(bin_data)
        cont_data = cont_data[keep]
        bin_data = bin_data[keep]

    g0 = bin_data == 0
    g1 = bin_data == 1

    x1 = cont_data[g1].mean()
    x0 = cont_data[g0].mean()
    sx = cont_data.std()
    p = np.sum(g1) / len(g1)
    q = 1 - p
    phi = norm.pdf(norm.ppf(q))

    bicorr = ((x1 - x0) * p * q) / (sx * phi)

    n1 = np.sum(g1)
    n0 = np.sum(g0)
    n = len(g0)

    bse = np.sqrt(((1 - bicorr ** 2) / n) * (1 / (p * q)))

    return bicorr, bse