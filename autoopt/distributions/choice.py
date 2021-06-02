"""
This module implements the different choice distributions.

A choice distribution is defined by a discrete target variable X, that can only take
some specified values (e.g. a list of strings).

There are two different distributions defined in this module:

    1. The "Choice"-Distribution defines an equal probability to each value.
    2. The "PChoice"-Distribution allows the user to weight each value differently.
"""
from .base import Distribution, _get_matplotlib


class WeightedChoice(Distribution):
    """
    Defines a parameter as a weighted choice.
    This parameter will sample each of the given
    choices according to the given weight.

    ..code:: python

        WeightedChoice(choices={"A": 0.5, "B": 0.25, "C": 0.25})

    The weights do not have to be a probabilities, but can be any number.
    The proportions of the weights define the probabilities.
    Thus, the definition "{'a': 1, 'b': 100}" defines, that it
    is 100 times more likely to choose 'b' than to choose 'a'.
    """

    def __init__(self, choices: dict):
        """
        Create a new probability choice parameter
        with `choices` as the possible values.
        The choices must be a dictionary containing the choice as key and
        the weight value.

        :param choices: A dictionary containing the value
                        of each choice as keys and the corresponding
                        weight as values.
        :type choices: dict[object, float]
        """
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

    def mean(self):
        probabilities = [
            weight / self.__weight_sum for weight in self.__choices.values()
        ]
        average = sum([i * prob for i, prob in enumerate(probabilities)])
        return list(self.__choices.keys())[int(round(average))]

    def pdf(self, x: object):
        if x not in self.__choices:
            return 0.0
        return self.__choices[x] / self.__weight_sum

    def _plot_min_value(self) -> float:  # pragma: no cover
        return 0.0

    def _plot_max_value(self) -> float:  # pragma: no cover
        return len(self.__choices)

    def _plot_label(self) -> str:  # pragma: no cover
        return ""

    def plot(self):
        plt = _get_matplotlib()
        if plt is None:
            return None
        figure = plt.figure()
        plt.ylabel("PDF(X)")
        plt.xlabel("X")
        x = range(0, len(self.__choices))
        y = [self.pdf(c) for c in self.__choices.keys()]
        plt.bar(x=x, height=y, tick_label=[str(c) for c in self.__choices.keys()])
        return figure


class Choice(WeightedChoice):
    """
    Defines a parameter as to be chosen from the given set of options.
    This parameter will be then chosen from this set during the optimization.
    Every choice has the same probability to be chosen.

    .. code:: python

        Choice(choices=["A", "B", "C"])
    """

    def __init__(self, choices: list):
        """
        Create a new choice parameter with `choices` as the possible values.

        :param choices: The possible values for this parameter.
        :type choices: List[object]
        """
        super().__init__(choices={c: 1 for c in choices})
