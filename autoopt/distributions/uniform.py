"""
This module implements the uniform distributions for parameter decoration.

It implements two different distributions:

    1. The Uniform distribution (Uniform)
    2. The quantized uniform distribution (QUniform)

The Uniform distribution describes a distribution between two values (min_value and max_value),
in which each point has an equal probability of being sampled.
Thus, the PDF of a Uniform distribution is defined as follows:

    P(X) = 1 / (max - min) if min <= X <= max else 0

The quantized uniform distribution instead, acts as a normal distribution, but the values
that get sampled, are quantized. Thus, the PDF of the QUniform distribution looks like this:

    P(X) = 1 / (max - min) if min <= round(X / q) <= max else 0
"""
from autoopt.distributions.base import Distribution, QMixin


class Uniform(Distribution):

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
        return self.__probability if self.min_value <= x <= self.max_value else 0

    def mean(self):
        return 0.5 * (self.min_value + self.max_value)

    @property
    def _plot_start(self):
        return self.min_value - 1

    @property
    def _plot_end(self):
        return self.max_value + 1


class QUniform(Uniform, QMixin):
    def __init__(self, parameter_name, min_value, max_value, q):
        super(QUniform, self).__init__(parameter_name=parameter_name, min_value=min_value, max_value=max_value)
        QMixin.__init__(self, q=q)

    def pdf(self, x):
        return super(QUniform, self).pdf(self.round_to_q(x))

    def mean(self):
        return self.round_to_q(super(QUniform, self).mean())

    @property
    def _plot_start(self):
        return self.round_to_q(self.min_value) - 2 * self.q

    @property
    def _plot_end(self):
        return self.round_to_q(self.max_value) + 2 * self.q
