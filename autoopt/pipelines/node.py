#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import abc
import logging

from autoopt.distributions.base import Distribution


class PipelineNode(object, metaclass=abc.ABCMeta):

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

    @abc.abstractmethod
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
        raise NotImplementedError("Has to be implemented by subclasses")

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

    @abc.abstractmethod
    def execute(self, input_data, **kwargs: dict):
        """
        Execute the algorithm this node implements on the given `input_data`.

        The `kwargs` passed to this are the values sampled from the
        `parameter_space` of this class.

        This method has to return the result of the algorithm which will
        get passed to the next node in the pipeline.

        :param input_data: The input data to execute the algorithm on.
        :type input_data: object
        :param kwargs: The parameters to initialize the algorithm with.
        :type kwargs: dict[str, object]
        :return: The result of the algorithm
        :rtype: object
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.name == self.name
        return False

    def __hash__(self):
        return hash(self.name)
