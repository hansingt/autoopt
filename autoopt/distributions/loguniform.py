"""
This module implements the logarithmic uniform distributions for parameter decoration.

It implements two different distributions:

    1. The logarithmic uniform distribution (LogUniform)
    2. The quantized logarithmic uniform distribution (QLogUniform)
"""
import logging

import numpy as np

from autoopt.distributions.base import QMixin
from autoopt.distributions.uniform import Uniform


class LogUniform(Uniform):
    """
    Defines a parameter as being drawn from the exponential of a uniform distribution.
    A log-uniform distributed parameter will be sampled from the exponential of equally
    distributed values between a given minimum and maximum value.
    The values for this parameter will be sampled from a function like:

        exp(uniform(min, max))

    This distribution causes that the logarithm of the samples values
    are being uniform distributed.

    This parameter is bound to [exp(min), exp(max)] and defined only for positive values.

    The PDF of this distribution is described by the following equation:

        P(X) = 1 / (X * (log(max_value) - log(min_value)) if 0 <= X and min_value <= X <= max_value else 0
    """
    def __init__(self, parameter_name, min_value, max_value):
        if min_value <= 0:
            raise ValueError("LogUniform distributions are only defined for positive intervals")
        super(LogUniform, self).__init__(parameter_name=parameter_name, min_value=min_value, max_value=max_value)
        self._min_log = np.log(min_value)
        self._max_log = np.log(max_value)

    def pdf(self, x):
        if x >= 0 and self.min_value <= x <= self.max_value:
            return 1 / (x * (self._max_log - self._min_log))
        return 0.0

    def mean(self):
        return (self.max_value - self.min_value) / (self._max_log - self._min_log)

    def plot(self):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            logging.getLogger().error("Error importing the matplotlib. "
                                      "Did you forget to install the 'plotting' extra?")
            return None
        start = self.min_value - 1
        stop = self.max_value + 1
        figure = plt.figure()
        plt.ylabel("PDF(X)")
        plt.xlabel("X")
        figure.suptitle("Logarithmic uniform distribution for parameter {self.name!s}".format(self=self))
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
        plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
        return figure


class QLogUniform(LogUniform, QMixin):
    """
    Defines a parameter as being drawn from the exponential
    of a uniform distribution and being bound to discrete values
    only.
    A q-log-uniform distributed parameter will be sampled from
    the exponential of equally distributed values
    between a given minimum and maximum value and will be bound by a
    regulation parameter.
    The values for this parameter will be sampled from a function like:

        round(exp(uniform(min, max)) / q) * q

    This distribution causes that the logarithm of the samples values
    are being uniform distributed.

    This parameter is bound to [exp(min), exp(max)] and defined only for positive values.
    """

    def __init__(self, parameter_name, min_value, max_value, q):
        super(QLogUniform, self).__init__(parameter_name=parameter_name, min_value=min_value, max_value=max_value)
        QMixin.__init__(self, q=q)

    def pdf(self, x):
        return super(QLogUniform, self).pdf(self.round_to_q(x))

    def mean(self):
        return self.round_to_q(super(QLogUniform, self).mean())

    def plot(self):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            logging.getLogger().error("Error importing the matplotlib. "
                                      "Did you forget to install the 'plotting' extra?")
            return None
        start = self.round_to_q(self.min_value) - 2 * self.q
        stop = self.round_to_q(self.max_value) + 2 * self.q
        figure = plt.figure()
        plt.ylabel("PDF(X)")
        plt.xlabel("X")
        figure.suptitle("Quantized logarithmic uniform distribution for parameter {self.name!s}".format(self=self))
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
        plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
        return figure

