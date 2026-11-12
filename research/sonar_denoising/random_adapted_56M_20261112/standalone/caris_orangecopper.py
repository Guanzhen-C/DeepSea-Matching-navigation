"""Python translation of the supplied caris_orangecopper.m control points."""
import numpy as np
from scipy.interpolate import PchipInterpolator


def caris_orangecopper(n=256):
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    cp = np.array([
        [0.0000, 0.0000, 0.0000],
        [0.2200, 0.1000, 0.0400],
        [0.4800, 0.2400, 0.0800],
        [0.7400, 0.3800, 0.1000],
        [0.9200, 0.5600, 0.1600],
        [1.0000, 0.7800, 0.3200],
        [1.0000, 0.9400, 0.7000],
    ])
    # MATLAB linspace(a,b,1) returns b, unlike NumPy.
    xi = np.array([1.]) if n == 1 else np.linspace(0., 1., int(n))
    return PchipInterpolator(np.linspace(0., 1., len(cp)), cp, axis=0)(xi)
