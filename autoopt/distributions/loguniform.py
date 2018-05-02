"""
This module implements the logarithmic uniform distributions for parameter decoration.

It implements two different distributions:

    1. The logarithmic uniform distribution (LogUniform)
    2. The quantized logarithmic uniform distribution (QLogUniform)
"""
import numpy as np
from autoopt.distributions.uniform import Uniform, QUniform


class LogUniform(Uniform):
    """
    Defines a parameter as being drawn from the exponential of a uniform distribution.
    A log-uniform distributed parameter will be sampled from the exponential of equally
    distributed values between a given minimum and maximum value.
    The values for this parameter will be sampled from a function like:

        exp(uniform(min, max))

    This distribution causes that the logarithm of the samples values
    are being uniform distributed.

    This parameter is bound to [exp(min), exp(max)].

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


class QLogUniform(QUniform, LogUniform):
    pass
