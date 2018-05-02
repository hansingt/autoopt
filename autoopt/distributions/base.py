"""
The base class for the different distribution decorators.

This module implements an abstract base class for the different
distributions defined in this package. Each distribution
can be used as a decorator, to decorate the hyperparameters
of the different algorithms.

    @Uniform("nu", 0, 1)
    def function_(nu):
        pass

    @Normal("gamma", 0, 1)
    class Algorithm(object):
        def __init__(gamma):
            pass

To help the algorithm experts define good default values,
each distribution can be plotted if the "plotting" extra
is installed. This plots the PDF (probability density function).

    Uniform("nu", 0, 1).plot()
"""
import abc
import copy
import logging
import numpy


class Distribution(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for creating distribution decorators.
    By decorating a parameters using a subclass of this, it is defined as
    a parameter to be optimized during the optimization process.

    BE CAREFUL WHEN IMPLEMENTING NEW DECORATORS. THEY HAVE TO BE SUPPORTED
    BY __ALL__ OPTIMIZATION ALGORITHMS
    """
    PARAMETER_CLASS_ATTRIBUTE = "_hyperparameters"

    def __init__(self, parameter_name):
        self._name = parameter_name

    @property
    def name(self):
        return self._name

    def plot(self):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            logging.getLogger().error("Error importing the matplotlib. "
                                      "Did you forget to install the 'plotting' extra?")
            return None
        figure = plt.figure()
        plt.ylabel("PDF(X)")
        plt.xlabel("X")
        figure.suptitle("Distribution for parameter {self.name!s}".format(self=self))
        # plot at least 1.000 points, 10.000 at most
        num_points = min(max(self._plot_end - self._plot_start, 1000), 10000)
        x = numpy.linspace(start=self._plot_start, stop=self._plot_end, num=num_points)
        y = numpy.vectorize(self.pdf)(x)
        axes = plt.plot(x, y)
        mean_x = self.mean()
        mean_y = self.pdf(mean_x)
        y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
        plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes[0].get_color(),
                    label="Mean: %g" % mean_x)
        plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
        return figure

    @abc.abstractmethod
    def pdf(self, x):
        """
        Calculate the probability for the given `x` to be chosen.

        :param x: The value to calculate the probability for.
        :return: The probability that the given value `x` gets sampled.
        :rtype: float
        """
        raise NotImplementedError("Has to be implemented by the subclasses")

    @abc.abstractmethod
    def mean(self):
        """
        Calculates the mean values of the distribution.

        :return: The mean value of the given distribution.
        :rtype: float
        """
        raise NotImplementedError("Has to be implemented by the subclasses")

    @property
    @abc.abstractmethod
    def _plot_start(self):
        return 0

    @property
    @abc.abstractmethod
    def _plot_end(self):
        return 1

    def __call__(self, obj):
        if not hasattr(obj, self.PARAMETER_CLASS_ATTRIBUTE):
            setattr(obj, self.PARAMETER_CLASS_ATTRIBUTE, set())
        # Copy the attribute to avoid overwriting inherited parameters.
        parameters = copy.copy(getattr(obj, self.PARAMETER_CLASS_ATTRIBUTE))
        if self in parameters:
            parameters.remove(self.name)
        parameters.add(self)
        setattr(obj, self.PARAMETER_CLASS_ATTRIBUTE, parameters)

    def __eq__(self, other):
        if hasattr(other, "name"):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{self.__class__.__name__}<{self.name}>".format(self=self)


class QMixin(object):
    """
    This mixin adds a regulation parameter `q`
    to bind a distribution to discrete values.
    """

    def __init__(self, q):
        """
        Add the new regulation parameter.

        :param q: The regulation value
        :type q: float
        """
        self.__q = q

    @property
    def q(self):
        return self.__q

    def round_to_q(self, value):
        """
        Round a value to the next multiple of `q`.
        This is required for the plotting only.

        :param value: The value to round
        :type value: float
        """
        return numpy.round(value / self.q) * self.q
