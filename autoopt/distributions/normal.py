#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
"""
This module implements the normal distributions for parameter decoration.

There are two different distributions implemented in this module:

    1. The normal distribution (Normal)
    2. The quantized normal distribution (QNormal)
"""
import numpy as np

from .base import Distribution, QMixin


class Normal(Distribution):
    """
    Defines a parameter as being drawn from a normal distribution.

    This distribution can be used if the values of a parameter should be sampled around
    a mean value (loc) with a specified standard deviation (scale).
    This distribution is unbound. Thus, theoretically all real values can be sampled,
    but the probability get's lower as the distance to the mean rises.

    The probability density function (PDF) is defined as follows:

        P(X) = 1/sqrt(2*pi*scale^2)*e^(-(X-loc)^2/(2*scale^2))
    """

    def __init__(self, loc: float, scale: float):
        self.__loc = loc
        self.__scale = scale

    @property
    def loc(self) -> float:
        """
        The mean value of the distribution
        """
        return self.__loc

    @property
    def scale(self) -> float:
        """
        The standard deviation of the distribution
        """
        return self.__scale

    def mean(self) -> float:
        """
        Calculate and return the mean value of the distribution

        :return: The mean value of this distribution
        """
        return self.loc

    def pdf(self, x: float) -> float:
        """
        Calculate the probability for the given `x` to be sampled.

        :param x: The value to calculate the probability for.
        :return: The probability that the given value is sampled.
        """
        return np.exp(-((x - self.loc) ** 2) / (2 * self.scale**2)) / (
            np.sqrt(2 * np.pi) * self.scale
        )

    def _plot_min_value(self):
        return self.mean() - 3 * self.scale

    def _plot_max_value(self):
        return self.mean() + 3 * self.scale

    def _plot_label(self):
        return "loc={self.loc:g}, scale={self.scale:g}".format(self=self)


class QNormal(Normal, QMixin):
    """
    The quantized normal distribution acts the same as the normal distribution,
    but the values, that get samples are quantized.
    Thus, it only samples values which are multiple of the given quantizer.

    The PDF of the QNormal distribution is defined as follows:

        P(X) = 1/sqrt(2*pi*scale^2)*e^(-(round(X/q)-loc)^2/(2*scale^2))
    """

    def __init__(self, loc: float, scale: float, q: float):
        super().__init__(loc=loc, scale=scale)
        QMixin.__init__(self, q=q)

    def mean(self) -> float:
        """
        Calculate and return the mean value of the distribution

        :return: The mean value of this distribution
        """
        return self.round_to_q(super().mean())

    def pdf(self, x) -> float:
        """
        Calculate the probability for the given `x` to be sampled.

        :param x: The value to calculate the probability for.
        :return: The probability that the given value is sampled.
        """
        return super().pdf(self.round_to_q(x))
