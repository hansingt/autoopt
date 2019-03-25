#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import copy
import inspect
import logging

import pkg_resources

from autoopt.distributions import Choice
from autoopt.distributions.base import Distribution
from autoopt.distributions.normal import Normal, QNormal


class PipelineNode(object):

    def __init__(self, node_name):
        """
        Creates a new node for a pipeline using the given node name.

        :param node_name: The name of the node to create a pipeline element for.
        :type node_name: str
        :return: A new pipeline element wrapping the given node.
        :rtype: PipelineNode
        """
        self.__name = node_name
        self.__class = None
        self.__parameters = None
        self.__optimization_parameters = None
        self.__logger = logging.getLogger("AutoOpt.Nodes.%s" % node_name)

    @property
    def name(self):
        return self.__name

    @property
    def node_class(self):
        if self.__class is None:
            for node in pkg_resources.iter_entry_points("autoopt.nodes", self.name):
                self.__class = node.load()
                break
            else:
                raise ValueError("No class found for node name '%s'. Please check your installation.")
        return self.__class

    @property
    def parameters(self):
        if self.__parameters is None:
            self.__parameters = set()
            for class_ in inspect.getmro(self.node_class):
                if class_ != object and hasattr(class_, "__init__"):
                    argspec = inspect.getfullargspec(class_.__init__)
                    if argspec.defaults is not None:
                        default_args = zip(argspec.args[-len(argspec.defaults):], argspec.defaults)
                        self.__parameters.update([arg for arg, default in default_args if arg != "self"])
        return self.__parameters

    @property
    def optimization_parameters(self):
        """
        Returns the names of the parameters of this node.
        If the class that implements this node defines it's hyper parameters, these will be used as
        optimization parameters. Otherwise every parameter of the node's __init__ method that has a default value
        is considered as a parameter of the node. In this case, the optimization parameters are the same as the
        `parameters`.

        :return: A list of the parameters of this node.
        :rtype: list[str]
        """
        if self.__optimization_parameters is None:
            if not hasattr(self.node_class, Distribution.PARAMETER_CLASS_ATTRIBUTE):
                self.__optimization_parameters = self.parameters
            else:
                self.__optimization_parameters = set([name for name in
                                                      getattr(self.node_class, Distribution.PARAMETER_CLASS_ATTRIBUTE)])
        return self.__optimization_parameters

    def parameter_space(self):
        """
        Returns a dictionary of all parameters of this node and their default values.
        If a parameter does not have a default, it is ignored and it is the responsibility of caller to ensure,
        that this parameter get's a value.
        This property should be overwritten in subclasses to create a real space from it, otherwise
        only the default values are used as the space of the parameters.

        :return: A dictionary containing all parameters and their default values.
        :rtype: dict[str, Distribution]
        """
        if not hasattr(self.node_class, Distribution.PARAMETER_CLASS_ATTRIBUTE):
            # Return 1 for every parameter not set
            space = set()
            parameters = self.optimization_parameters
            for class_ in inspect.getmro(self.node_class):
                if class_ != object and hasattr(class_, "__init__"):
                    argspec = inspect.getfullargspec(class_.__init__)
                    if argspec.defaults is not None:
                        default_args = zip(argspec.args[-len(argspec.defaults):], argspec.defaults)
                        for param, default in default_args:
                            if param in parameters:
                                if isinstance(default, bool):
                                    # Add a boolean choice
                                    space.add(Choice(param, choices=[True, False]))
                                elif isinstance(default, float):
                                    # Add a normal distribution
                                    space.add(Normal(param, loc=default, scale=1.0))
                                elif isinstance(default, int):
                                    # Add a Q-Normal distribution
                                    space.add(QNormal(param, loc=default, scale=1.0, q=1))
                                else:
                                    space.add(Choice(param, choices=[default]))
        else:
            space = getattr(self.node_class, Distribution.PARAMETER_CLASS_ATTRIBUTE).values()
        # Create a dictionary containing the optimization name as a key
        return {self._make_parameter_name(parameter.name): parameter for parameter in space}

    def _make_parameter_name(self, parameter):
        """
        Creates an unique name for the given parameter.
        This method uses the scheme __{node_name}_{parameter}__ to make each parameter a unique variable.

        :param parameter: The name of the parameter to make unique
        :type parameter: str
        :return: A unique name for the given parameter.
        :rtype: str
        """
        return "{node_name}_{parameter}".format(
            node_name=self.name,
            parameter=parameter
        )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.name == self.name
        return False

    def __hash__(self):
        return hash(self.name)
