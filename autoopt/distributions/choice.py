"""
This module implements the different choice distributions.

A choice distribution is defined by a discrete target variable X, that can only take
some specified values (e.g. a list of strings). There are two different distributions defined in this module:

    1. The "Choice"-Distribution defines an equal probability to each value.
    2. The "PChoice"-Distribution allows the user to weight each value differently.
"""
import logging

from autoopt.distributions.base import Distribution


class WeightedChoice(Distribution):
    """
    Defines a parameter as a weighted choice.
    This parameter will sample each of the given
    choices according to the given weight.

    >>> @WeightedChoice("test", choices={"A": 0.5, "B": 0.25, "C": 0.25})
    ... def fun(test):
    ...     pass

    The weight does not have to be a probability, but can be any number.
    The proportions of the weights define the probabilities.
    Thus, the definition "{'a': 1, 'b': 100}" defines, that it
    is 100 times more likely to choose 'b' than to choose 'a'.
    """

    def __init__(self, parameter_name, choices):
        """
        Create a new probability choice parameter
        with `parameter_name` as name and `choices`
        as the possible values.
        Each choice must be a tuple containing first
        the probability in range from 0 to 1 for that
        choice and the value to choose as a second argument.
        The probabilities of all choices need to sum up to 1.

        :param parameter_name: The name of the parameter to create.
        :type parameter_name: str
        :param choices: A dictionary of tuples containing the value
                        of each choice as keys and the corresponding
                        probabilities as values.
        :type choices: dict[object, float]
        """
        super(WeightedChoice, self).__init__(parameter_name)
        self.__choices = choices
        self.__weight_sum = sum(choices.values())

    @property
    def choices(self):
        """
        Returns a dictionary containing the choice values as keys
        and the weight for each of them as values.

        :return: A list of values and probabilities.
        :rtype: dict[object, float]
        """
        return self.__choices.copy()

    def pdf(self, x):
        if x not in self.__choices:
            return 0.0
        else:
            return self.__choices[x] / self.__weight_sum

    def plot(self):
        try:
            from matplotlib import pyplot as plt
        except ImportError:  # pragma: no cover
            logging.getLogger().error("Error importing the matplotlib. "
                                      "Did you forget to install the 'plotting' extra?")
            return None
        figure = plt.figure()
        plt.ylabel("PDF(X)")
        plt.xlabel("X")
        figure.suptitle("Probability choice distribution for parameter {self.name!s}".format(self=self))
        x = range(0, len(self.__choices))
        y = [self.pdf(c) for c in self.__choices.keys()]
        plt.bar(x=x, height=y, tick_label=[str(c) for c in self.__choices.keys()])
        return figure


class Choice(WeightedChoice):
    """
    Defines a parameter as to be chosen from
    the given set of options.
    This parameter will be then chosen from this
    set during the optimization.

    >>> @Choice("test", choices=["A", "B", "C"])
    ... def fun(test):
    ...     pass
    """

    def __init__(self, parameter_name, choices):
        """
        Create a new choice parameter with `parameter_name` as name
        and `choices` as the possible values.

        :param parameter_name: The name of the parameter to create.
        :type parameter_name: str
        :param choices: The possible values for this parameter.
        :type choices: str | List[str]
        """
        if not isinstance(choices, list):
            choices = [choices]
        super(Choice, self).__init__(parameter_name=parameter_name, choices={c: 1 for c in choices})

    @property
    def choices(self):
        """
        Returns the list of values to choose from

        :return: A list of values.
        :rtype: list[object]
        """
        return super(Choice, self).choices.keys()
