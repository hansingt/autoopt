#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import abc
import logging
from typing import Dict, Any

from autoopt.distributions.base import Distribution


class PipelineNode(object, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns the name of this node.

        Typically, this is the name of the algorithm implemented by this pipeline node.

        :return: The name of this node
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    @abc.abstractmethod
    def parameter_space(self) -> Dict[str, Distribution]:
        """
        Returns a dictionary of all parameters of this node and their default values.
        If a parameter does not have a default, it is ignored and it is the responsibility of caller to ensure,
        that this parameter get's a value.
        This property should be overwritten in subclasses to create a real space from it, otherwise
        only the default values are used as the space of the parameters.

        :return: A dictionary containing all parameters and their default values.
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    @property
    @abc.abstractmethod
    def input_type(self) -> str:
        """
        The data type this node requests the input to be in.

        The data type can be any type, but it should be as general
        as possible to make the node being usable in as many pipelines as possible.

        :return: The data type of the input data
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    @property
    @abc.abstractmethod
    def output_type(self) -> str:
        """
        The type of the data this node returns

        The data type can be any type, but it should be as general
        as possible to make the node being usable in as many pipelines as possible.

        :return: The type of the output data
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    @abc.abstractmethod
    def execute(self, input_data: Any, **kwargs: Dict[str, Any]) -> Any:
        """
        Execute the algorithm this node implements on the given `input_data`.

        The type of the input data is determined by the `input_data_type` property
        and the pipeline generator assures, that these types match.

        The `kwargs` passed to this are the values sampled from the
        `parameter_space` of this class.

        This method has to return the result of the algorithm which will
        get passed to the next node in the pipeline. The type of the result
        has to match the type defined by the `output_data_type` property.

        :param input_data: The input data to execute the algorithm on.
        :param kwargs: The parameters to initialize the algorithm with.
        :return: The result of the algorithm
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PipelineNode):
            return other.name == self.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)
