from setuptools_scm import meta


# -*- coding: utf-8 -*-
r"""
Sweep module. Current approach to the problem (hopefully, the last)
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2022, All rights reserved."
__maintainer__ = "Stefan Ulbrich"
__email__ = "stefan.frank.ulbrich@gmail.com"
__date__ = "2022-04-03"
__version__ = "0.1"
__status__ = "alpha"
# __all__ = ["create_catmull_rom", "create_nonuniform"]

import numpy as np
import re
from scipy.interpolate import CubicHermiteSpline


def parse_input(s: str) -> np.ndarray:
    return np.asarray(re.findall("-?\d+\.\d+", s)).astype(float).reshape(-1, 3)


def create_catmull_rom(x, y, T=0.5, axis=0, extrapolate=None):
    return CubicHermiteSpline(
        x,
        y,
        np.vstack(
            (
                (y[1] - y[0]) / (x[1] - x[0]),
                T * 2 * (y[2:] - y[:-2]) / (x[2:] - x[:-2]).reshape(-1, 1),
                (y[-1] - y[-2]) / (x[-1] - x[-2]),
            )
        ),
        axis=axis,
        extrapolate=extrapolate,
    )


def create_nonuniform(x, y, t="centripetal", axis=0, extrapolate=None):
    p = 0.25 if t == "centripetal" else 0.5

    d = (((a[1:] - a[:-1]) ** 2).sum(axis=1)) ** p

    raise NotImplementedError
    # let t1 = ( x1 - x0 ) / dt0 - ( x2 - x0 ) / ( dt0 + dt1 ) + ( x2 - x1 ) / dt1;
    # let t2 = ( x2 - x1 ) / dt1 - ( x3 - x1 ) / ( dt1 + dt2 ) + ( x3 - x2 ) / dt2;

    # // rescale tangents for parametrization in [0,1]
    # t1 *= dt1;
    # t2 *= dt1;
    return CubicHermiteSpline(
        x,
        y,
        np.vstack((y[1] - y[0], T * y[2:] - y[:-2], y[3] - y[2])),
        axis=axis,
        extrapolate=extrapolate,
    )