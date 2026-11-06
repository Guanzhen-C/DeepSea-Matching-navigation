"""Literal translation of the supplied Bathymetry_SonarDefaultColormap3D.m."""
import numpy as np


def Bathymetry_SonarDefaultColormap3D(n=256):
    # The supplied MATLAB function always outputs 256 rows, even if n differs.
    # Preserve that behavior rather than silently correcting the original.
    cp = np.array([
        [1.00, 0.20, 0.00],
        [1.00, 0.80, 0.00],
        [0.20, 0.80, 0.20],
        [0.00, 0.70, 0.80],
        [0.00, 0.20, 0.80],
        [0.00, 0.00, 0.60],
    ])
    x = np.linspace(0., 1., len(cp))
    xi = np.linspace(0., 1., 256)
    return np.column_stack([np.interp(xi, x, cp[:, i]) for i in range(3)])
