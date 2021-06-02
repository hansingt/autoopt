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
import logging

import numpy


def _get_matplotlib():
    try:
        from matplotlib import pyplot as plt

        return plt
    except ImportError:  # pragma: no cover
        logging.getLogger().error(
            "Error importing the matplotlib. "
            "Did you forget to install the 'plotting' extra?"
        )
        return None


class Distribution:
    """
    Abstract base class for creating distribution decorators.
    By decorating a parameters using a subclass of this, it is defined as
    a parameter to be optimized during the optimization process.

    BE CAREFUL WHEN IMPLEMENTING NEW DECORATORS. THEY HAVE TO BE SUPPORTED
    BY __ALL__ OPTIMIZATION ALGORITHMS
    """

    @abc.abstractmethod
    def mean(self) -> float:
        """
        Return the mean of this distribution.

        The mean value is the mean of all samples of this distribution
        if it would get sampled infinite often.

        :return: The mean of this distribution.
        """
        raise NotImplementedError("Has to be implemented by the subclasses")

    @abc.abstractmethod
    def pdf(self, x: object) -> float:
        """
        Calculate the probability for a given value `x` to be sampled
        in this distribution.

        The probability has to be in the range [0, 1].

        :param x: The value to check the probability for
        :return: The probability, that the given value `x` get's sampled
        """
        raise NotImplementedError("Has to be implemented by the subclasses")

    @abc.abstractmethod
    def _plot_min_value(self) -> float:
        """
        The minium value to start the plot with.
        """
        raise NotImplementedError("Has to be implemented by the subclasses")

    @abc.abstractmethod
    def _plot_max_value(self) -> float:
        """
        The maximum value to stop the plot with.
        """
        raise NotImplementedError("Has to be implemented by the subclasses")

    @abc.abstractmethod
    def _plot_label(self) -> str:
        """
        The label to display at the plot
        """
        raise NotImplementedError("Has to be implemented by the subclasses")

    def plot(self):
        """
        Create a plot to visualize the PDF of this
        distribution and return the figure created.

        :return: ``None`` if matplotlib is not installed, otherwise
                 the figure which contains the plot of the PDF of this
                 distribution.
        """
        plt = _get_matplotlib()
        if plt is None:
            return None
        start = self._plot_min_value()
        stop = self._plot_max_value()
        figure = plt.figure()
        plt.ylabel("PDF(X)")
        plt.xlabel("X")
        # plot at least 1.000 points, 10.000 at most
        num_points = int(min(max(stop - start, 1000), 10000))
        x = numpy.linspace(start=start, stop=stop, num=num_points)
        y = numpy.vectorize(self.pdf)(x)
        axes = plt.plot(x, y, label=self._plot_label())
        mean_x = self.mean()
        mean_y = self.pdf(mean_x)
        y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
        plt.axvline(
            x=mean_x,
            ymax=y_max,
            linestyle="--",
            color=axes[0].get_color(),
            label="Mean: %g" % mean_x,
        )
        legend = plt.legend(loc="best", fancybox=True, framealpha=0.2)
        legend.set_draggable(True)
        return figure


class QMixin:
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
    def q(self) -> float:
        """
        Return the requlation value of this distribution.

        :return: The requlation value of this distribution
        """
        return self.__q

    def round_to_q(self, value: float) -> float:
        """
        Round a value to the next multiple of `q`.

        :param value: The value to round
        :return: The quantized value which is the nearest multiple of q
        """
        return numpy.round(value / self.q) * self.q
