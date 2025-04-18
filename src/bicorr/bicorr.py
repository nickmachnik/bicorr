"""
Computation of biserial and tetrachoric correlations in python.
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
        (float float): correlation, standard error
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
    
    n = len(g0)
    # Hunter and Schmidt 1990
    bse = np.sqrt(((p * q) / phi ** 2) * (1 - bicorr ** 2) ** 2 / (n - 3))
    return bicorr, bse


def tetrachoric_correlation(x: np.array, y: np.array, method='gamma') -> (float, float):
    """Compute tetrachoric correlation and its standard error.

    Args:
        x (array like): data of first binary variable
        y (array like): data of second binary variable
        method (str, optional): Method to use, one of {'gamma', 'inverse-cosine'}. Defaults to 'gamma'.

    Raises:
        ValueError: When the number of (1, 1) observations is 0.
        ValueError: When the specified method is unknown.

    Returns:
        (float, float): correlation, standard error
    """
    # Remove missing values
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]

    # Contingency table
    n11 = np.sum((x == 1) & (y == 1))
    n10 = np.sum((x == 1) & (y == 0))
    n01 = np.sum((x == 0) & (y == 1))
    n00 = np.sum((x == 0) & (y == 0))
    n = n11 + n10 + n01 + n00

    if n11 == 0:
        raise ValueError("Number of (1, 1) observations is 0.")

    if method == 'gamma':
        gamma = ((n11 * n00) / (n01 * n10)) ** (np.pi / 4)
        rtet = (gamma - 1) / (gamma + 1)
        se_part1 = (np.pi * gamma / (2 * (gamma + 1) ** 2)) ** 2 
        se_part2 = (1 / n11 + 1 / n10 + 1 / n01 + 1 / n00)
        se = np.sqrt(se_part1 * se_part2)
    elif method == 'inverse-cosine':
        sig = 1 + np.sqrt((n11 * n00) / (n01 * n10))
        rtet = np.cos(np.pi / sig)
        p0 = (n11 + n01) / n
        p1 = (n11 + n10) / n
        h0 = norm.pdf(norm.ppf(p0))
        h1 = norm.pdf(norm.ppf(p1))
        se = np.sqrt((p0 * (1 - p0) * p1 * (1 - p1)) / (n * (h0 * h1) ** 2))
    else:
        raise ValueError("Method has to be in {'gamma', 'inverse-cosine'}")

    return rtet, se