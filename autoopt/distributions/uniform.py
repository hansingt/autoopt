"""
This module implements the uniform distributions for parameter decoration.

It implements two different distributions:

    1. The Uniform distribution (Uniform)
    2. The quantized uniform distribution (QUniform)
"""
import logging

import numpy as np

from autoopt.distributions.base import Distribution, QMixin


class Uniform(Distribution):
    """
    The Uniform distribution describes a distribution between two values (min_value and max_value),
    in which each point has an equal probability of being sampled.
    Thus, the PDF of a Uniform distribution is defined as follows:

    P(X) = 1 / (max - min) if min <= X <= max else 0
    """
    def __init__(self, parameter_name, min_value, max_value):
        if min_value >= max_value:
            raise ValueError("The minimum value has to be smaller than the maximum value %g >= %g" %
                             (min_value, max_value))
        super(Uniform, self).__init__(parameter_name=parameter_name)
        self._min_value = min_value
        self._max_value = max_value
        self.__probability = 1 / (max_value - min_value)

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    def pdf(self, x):
        """
        Calculate the probability for the given `x` to be chosen.

        :param x: The value to calculate the probability for.
        :return: The probability that the given value `x` gets sampled.
        :rtype: float
        """
        return self.__probability if self.min_value <= x <= self.max_value else 0

    def mean(self):
        """
        Calculates the mean values of the distribution.

        :return: The mean value of the given distribution.
        :rtype: float
        """
        return 0.5 * (self.min_value + self.max_value)

    def plot(self):
        try:
            from matplotlib import pyplot as plt
        except ImportError:  # pragma: no cover
            logging.getLogger().error("Error importing the matplotlib. "
                                      "Did you forget to install the 'plotting' extra?")
            return None
        start = self.min_value - 1
        stop = self.max_value + 1
        figure = plt.figure()
        plt.ylabel("PDF(X)")
        plt.xlabel("X")
        figure.suptitle("Uniform distribution for parameter {self.name!s}".format(self=self))
        # plot at least 1.000 points, 10.000 at most
        num_points = min(max(stop - start, 1000), 10000)
        x = np.linspace(start=start, stop=stop, num=num_points)
        y = np.vectorize(self.pdf)(x)
        axes = plt.plot(x, y, label="min={self.min_value:g}, max={self.max_value:g}".format(self=self))
        mean_x = self.mean()
        mean_y = self.pdf(mean_x)
        y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
        plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes[0].get_color(),
                    label="Mean: %g" % mean_x)
        legend = plt.legend(loc="best", fancybox=True, framealpha=0.2)
        legend.set_draggable(True)
        return figure


class QUniform(Uniform, QMixin):
    """
    The quantized uniform distribution instead, acts as a normal distribution, but the values
    that get sampled, are quantized. Thus, the PDF of the QUniform distribution looks like this:

    P(X) = 1 / (max - min) if min <= round(X / q) <= max else 0
    """
    def __init__(self, parameter_name, min_value, max_value, q):
        super(QUniform, self).__init__(parameter_name=parameter_name, min_value=min_value, max_value=max_value)
        QMixin.__init__(self, q=q)

    def pdf(self, x):
        return super(QUniform, self).pdf(self.round_to_q(x))

    def mean(self):
        return self.round_to_q(super(QUniform, self).mean())

    def plot(self):
        try:
            from matplotlib import pyplot as plt
        except ImportError:  # pragma: no cover
            logging.getLogger().error("Error importing the matplotlib. "
                                      "Did you forget to install the 'plotting' extra?")
            return None
        start = self.round_to_q(self.min_value) - 2 * self.q
        stop = self.round_to_q(self.max_value) + 2 * self.q
        figure = plt.figure()
        plt.ylabel("PDF(X)")
        plt.xlabel("X")
        figure.suptitle("Quantized uniform distribution for parameter {self.name!s}".format(self=self))
        # plot at least 1.000 points, 10.000 at most
        num_points = min(max(stop - start, 1000), 10000)
        x = np.linspace(start=start, stop=stop, num=num_points)
        y = np.vectorize(self.pdf)(x)
        axes = plt.plot(x, y, label="min={self.min_value:g}, max={self.max_value:g}, q={self.q:g}".format(self=self))
        mean_x = self.mean()
        mean_y = self.pdf(mean_x)
        y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
        plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes[0].get_color(),
                    label="Mean: %g" % mean_x)
        legend = plt.legend(loc="best", fancybox=True, framealpha=0.2)
        legend.set_draggable(True)
        return figure


