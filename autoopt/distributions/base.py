"""
The base class for the different distribution decorators.

This module implements an abstract base class for the different
distributions defined in this package. Each distribution
can be used to declare the hyperparameters
of the different algorithms.

To help the algorithm experts define good default values,
each distribution can be plotted if the "plotting" extra
is installed. This plots the PDF (probability density function).

.. code:: python

    Uniform(0, 1).plot()
"""
import abc

import numpy


class Distribution(object):
    """
    Abstract base class for creating distribution decorators.
    By decorating a parameters using a subclass of this, it is defined as
    a parameter to be optimized during the optimization process.

    BE CAREFUL WHEN IMPLEMENTING NEW DECORATORS. THEY HAVE TO BE SUPPORTED
    BY __ALL__ OPTIMIZATION ALGORITHMS
    """
    @abc.abstractmethod
    def mean(self):
        """
        Return the mean of this distribution.

        The mean value is the mean of all samples of this distribution
        if it would get sampled infinite often.

        :return: The mean of this distribution.
        :rtype: object
        """
        raise NotImplementedError("Has to be implemented by the subclasses")

    @abc.abstractmethod
    def pdf(self, x: object):
        """
        Calculate the probability for a given value `x` to be sampled
        in this distribution.

        The probability has to be in the range [0, 1].

        :param x: The value to check the probability for
        :return: The probability, that the given value `x` get's sampled
        :rtype: float
        """
        raise NotImplementedError("Has to be implemented by the subclasses")

    @abc.abstractmethod
    def plot(self):
        raise NotImplementedError("Has to be implemented by the subclasses")


class QMixin(object):
    """
    This mixin adds a regulation parameter `q`
    to bind a distribution to discrete values.
    """

    def __init__(self, q: float):
        """
        Add the new regulation parameter.

        :param q: The regulation value
        """
        self.__q = q

    @property
    def q(self):
        return self.__q

    def round_to_q(self, value: float):
        """
        Round a value to the next multiple of `q`.

        :param value: The value to round
        :return: The quantized value which is the nearest multiple of q
        :rtype: float
        """
        return numpy.round(value / self.q) * self.q
