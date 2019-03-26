#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import numpy as np
import random

from autoopt.distributions.normal import Normal


def create_dist(loc=None, scale=None):
    loc = loc if loc is not None else 0
    scale = scale if scale is not None else 1
    return Normal(loc=loc, scale=scale)


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
    dist = create_dist(loc=loc, scale=scale)
    x = np.linspace(start=loc - 2 * scale, stop=loc + 2 * scale, num=1000)
    y = 1. / np.sqrt(2 * np.pi * scale ** 2) * np.exp(- (x - loc) ** 2 / (2 * scale ** 2))
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
