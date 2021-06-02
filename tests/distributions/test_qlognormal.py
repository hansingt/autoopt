#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import random

import numpy as np

from autoopt.distributions import QLogNormal


def create_dist(loc=None, scale=None, q=None):
    loc = loc if loc is not None else 0
    scale = scale if scale is not None else 1
    q = q if q is not None else 1
    return QLogNormal(loc=loc, scale=scale, q=q)


def test_loc():
    loc = random.random()
    dist = create_dist(loc=loc)
    assert dist.loc == loc


def test_scale():
    scale = random.random()
    dist = create_dist(scale=scale)
    assert dist.scale == scale


def test_pdf():
    loc = 0
    scale = 1.0
    q = 0.5
    dist = create_dist(loc=loc, scale=scale, q=q)

    def check_pdf(x_):
        if x_ <= 0:
            return 0.0
        else:
            return (
                1.0
                / (np.sqrt(2 * np.pi) * scale * x_)
                * np.exp(-((np.log(x_) - loc) ** 2) / (2 * scale ** 2))
            )

    start = dist.mean() - 2 * scale
    stop = dist.mean() + 2 * scale
    x = np.vectorize(dist.round_to_q)(np.linspace(start=start, stop=stop, num=1000))
    y = np.vectorize(check_pdf)(x)
    assert np.allclose(y, np.vectorize(dist.pdf)(x))


def test_plot():
    loc = 0
    scale = 1.0
    dist = create_dist(loc=loc, scale=scale)
    plot = dist.plot()
    try:
        from matplotlib import pyplot as plt

        assert isinstance(plot, plt.figure().__class__)
    except ImportError:
        assert plot is None
