"""
This module implements the uniform distributions for parameter decoration.

It implements two different distributions:

    1. The Uniform distribution (Uniform)
    2. The quantized uniform distribution (QUniform)
"""
from .base import Distribution, QMixin


class Uniform(Distribution):
    """
    The Uniform distribution describes a distribution between two values
    (min_value and max_value), in which each point has an equal probability
    of being sampled.
    Thus, the PDF of a Uniform distribution is defined as follows:

        P(X) = 1 / (max - min) if min <= X <= max else 0
    """

    def __init__(self, min_value: float, max_value: float):
        if min_value >= max_value:
            raise ValueError(
                "The minimum value has to be smaller than the maximum value %g >= %g"
                % (min_value, max_value)
            )
        self._min_value = min_value
        self._max_value = max_value
        self.__probability = 1 / (max_value - min_value)

    @property
    def min_value(self) -> float:
        """
        The minimum value of this distribution.

        This is the smalest value which can be sampled.
        """
        return self._min_value

    @property
    def max_value(self) -> float:
        """
        The maximum value of this distribution.

        This is the largest value which can be sampled.
        """
        return self._max_value

    def pdf(self, x: float):
        return self.__probability if self.min_value <= x <= self.max_value else 0

    def mean(self):
        return 0.5 * (self.min_value + self.max_value)

    def _plot_min_value(self):
        return self.min_value - 1

    def _plot_max_value(self):
        return self.max_value + 1

    def _plot_label(self):
        return "min={self.min_value:g}, max={self.max_value:g}".format(self=self)


class QUniform(Uniform, QMixin):
    """
    The quantized uniform distribution instead, acts as a normal distribution,
    but the values that get sampled, are quantized.

    Thus, the PDF of the QUniform distribution looks like this:

        P(X) = 1 / (max - min) if min <= round(X / q) <= max else 0
    """

    def __init__(self, min_value: float, max_value: float, q: float):
        super().__init__(min_value=min_value, max_value=max_value)
        QMixin.__init__(self, q=q)

    def pdf(self, x: float):
        return super().pdf(self.round_to_q(x))

    def mean(self):
        return self.round_to_q(super().mean())

    def _plot_min_value(self):
        return self.round_to_q(self.min_value) - 2 * self.q

    def _plot_max_value(self):
        return self.round_to_q(self.max_value) + 2 * self.q

    def _plot_label(self):
        return "min={min_value:g}, max={max_value:g}, q={q:g}".format(
            min_value=self.min_value,
            max_value=self.max_value,
            q=self.q,
        )
