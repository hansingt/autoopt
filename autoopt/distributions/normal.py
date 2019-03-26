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
import logging
import numpy as np

from .base import Distribution, QMixin


class Normal(Distribution):
    """
    Defines a parameter as being drawn from a normal distribution.

    This distribution can be used if the values of a parameter should be sampled around
    a mean value (loc) with a specified standard deviation (scale). This distribution is unbound.
    Thus, theoretically all real values can be sampled, but the probability get's lower as the distance
    to the mean rises.

    The probability density function (PDF) is defined as follows:

    ..math::

        P(X) = 1 / sqrt(2 * pi * scale ** 2) * e ** (- (X - loc) ** 2 / (2 * scale ** 2))
    """
    def __init__(self, loc: float, scale: float):
        self.__loc = loc
        self.__scale = scale

    @property
    def loc(self):
        return self.__loc

    @property
    def scale(self):
        return self.__scale

    def mean(self):
        return self.loc

    def pdf(self, x: float):
        """
        Calculate the probability for the given `x` to be sampled.

        :param x: The value to calculate the probability for.
        :return: The probability that the given value is sampled.
        :rtype: float
        """
        return np.exp(-(x - self.loc) ** 2 / (2 * self.scale ** 2)) / (np.sqrt(2 * np.pi) * self.scale)

    def plot(self):
        try:
            from matplotlib import pyplot as plt
        except ImportError:  # pragma: no cover
            logging.getLogger().error("Error importing the matplotlib. "
                                      "Did you forget to install the 'plotting' extra?")
            return None
        # 3 * scale ~ 99% of the values
        start = self.mean() - 3 * self.scale
        stop = self.mean() + 3 * self.scale
        figure = plt.figure()
        plt.ylabel("PDF(X)")
        plt.xlabel("X")
        # plot at least 1.000 points, max 10.000
        num_points = int(min(max(stop - start, 1000), 10000))
        x = np.linspace(start=start, stop=stop, num=num_points)
        y = np.vectorize(self.pdf)(x)
        axes = plt.plot(x, y, label="loc={self.loc:g}, scale={self.scale:g}".format(self=self))
        mean_x = self.mean()
        mean_y = self.pdf(mean_x)
        y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
        plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes[0].get_color(),
                    label="Mean: %g" % mean_x)
        legend = plt.legend(loc="best", fancybox=True, framealpha=0.2)
        legend.set_draggable(True)
        return figure


class QNormal(Normal, QMixin):
    """
    The quantized normal distribution acts the same as the normal distribution, but the values,
    that get samples are quantized. Thus, it only samples values which are multiple of the given quantizer.

    The PDF of the QNormal distribution is defined as follows:

    P(X) = 1 / sqrt(2 * pi * scale ** 2) * e ** (- (round(X / q) - loc) ** 2 / (2 * scale ** 2))
    """
    def __init__(self, loc: float, scale: float, q: float):
        super(QNormal, self).__init__(loc=loc, scale=scale)
        QMixin.__init__(self, q=q)

    def mean(self):
        return self.round_to_q(super(QNormal, self).mean())

    def pdf(self, x):
        return super(QNormal, self).pdf(self.round_to_q(x))
