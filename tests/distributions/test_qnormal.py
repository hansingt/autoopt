#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import numpy as np
import random

from autoopt.distributions.normal import QNormal


def create_dist(loc=None, scale=None, q=None):
    loc = loc if loc is not None else 0
    scale = scale if scale is not None else 1
    q = q if q is not None else 1
    return QNormal("test", loc=loc, scale=scale, q=q)


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
    start = dist.round_to_q(loc - 2 * scale)
    stop = dist.round_to_q(loc + 2 * scale)
    x = np.linspace(start=start, stop=stop, num=1000)
    y = 1. / np.sqrt(2 * np.pi * scale ** 2) * np.exp(- (dist.round_to_q(x) - loc) ** 2 / (2 * scale ** 2))
    assert all(y == dist.pdf(x))


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
