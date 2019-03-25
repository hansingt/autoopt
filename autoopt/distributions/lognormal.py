#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
"""
This module implements the logarithmic normal distributions for parameter decoration.

There are two different distributions implemented in this module:

    1. The logarithmic normal distribution (LogNormal)
    2. The quantized logarithmic normal distribution (QLogNormal)
"""

import numpy as np

from autoopt.distributions import Normal
from .base import QMixin


class LogNormal(Normal):
    """
    Defines a parameter as being drawn from a logarithmic normal distribution.

    This distribution can be used if the values of a parameter should be sampled around the logarithm of
    a mean value (loc) with a specified standard deviation (scale). The `loc` and `scale` parameters do not
    define the mean and standard deviation of the logarithmic normal distribution, but instead of the underlying
    normal distribution. Thus, a random variable `X` is defined as being logarithmic normal distributed if
    a second variable `Y` exists with :math:`Y = normal(loc, scale)` such that :math:`X = e^Y`.

    For positive values, this distribution is unbound. Thus, theoretically all positive real values can be sampled,
    but the probability get's lower as the distance to the mean rises. For negative values, this distribution is
    undefined and thus, they can't be sampled.

    The probability density function (PDF) is defined as follows:

        P(X) = 1 / sqrt(2 * pi * scale ** 2) * e ** (- (ln(X) - loc) ** 2 / (2 * scale ** 2)) if X > 0 else 0
    """
    def mean(self):
        return np.exp(self.loc + self.scale ** 2 / 2)

    def pdf(self, x):
        """
        Calculate the probability for the given `x` to be sampled.

        :param x: The value to calculate the probability for.
        :type x: np.ndarray
        :return: The probability that the given value is sampled.
        :rtype: np.ndarray
        """
        return np.vectorize(lambda x_: super(LogNormal, self).pdf(np.log(x_)) / x_ if x_ > 0 else 0.0)(x)


class QLogNormal(LogNormal, QMixin):
    """
    The quantized normal distribution acts the same as the normal distribution, but the values,
    that get samples are quantized. Thus, it only samples values which are multiple of the given quantizer.

    The PDF of the QNormal distribution is defined as follows:

    P(X) = 1 / sqrt(2 * pi * scale ** 2) * e ** (- (round(X / q) - loc) ** 2 / (2 * scale ** 2))
    """
    def __init__(self, parameter_name, loc, scale, q=1.0):
        super(QLogNormal, self).__init__(parameter_name=parameter_name, loc=loc, scale=scale)
        QMixin.__init__(self, q=q)

    def mean(self):
        return self.round_to_q(super(QLogNormal, self).mean())

    def pdf(self, x):
        return super(QLogNormal, self).pdf(self.round_to_q(x))
