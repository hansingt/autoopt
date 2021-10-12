#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
"""
This module implements a processing pipeline.

These pipelines contain different nodes which have
an input and output type and a processing algorithm to
convert data from the input into the output format.
The input of each node must match the ouput of the previous node.

These pipelines can then be executed to process the input data
and calculate a loss. This loss will then be used to optimize the
parameters of the algorithms in the pipeline.
"""
from typing import Dict, Any

from autoopt.distributions.base import Distribution
from .node import PipelineNode


class Pipeline:
    """
    Implements a processing pipeline.

    It contains different nodes which have an input and output type
    and a processing algorithm to convert data from the input into
    the output format.
    The input of each node must match the ouput of the previous node.

    It can then be executed to process the input data and calculate
    a loss. This loss will then be used to optimize the parameters
    of the algorithms in this pipeline.
    """

    def __init__(self):
        self.__nodes = []

    def add_node(self, node: PipelineNode):
        """
        Add a new node to this pipeline.

        This assures, that the input data type of the node matches
        the current output data type of the pipeline to make sure,
        that the input data passed in later can be passed along the
        pipeline.

        :param node: The node to add to this pipeline.
        """
        if self.output_type not in ("Any", node.input_type):
            raise ValueError(
                f"Can't add node '{node.name}': "
                "The input types don't match. "
                f"Expected '{self.output_type}' but got '{node.input_type}'"
            )
        self.__nodes.append(node)

    def parameter_space(self) -> Dict[str, Dict[str, Distribution]]:
        """
        Returns the parameter space of the pipeline.

        The parameter space of the pipeline is defined as the
        parameter spaces of all nodes. Thus, this method returns
        a dictionary containing the node name as key an the parameter
        space of the node as key.

        :return: The parameter space of the whole pipeline
        """
        return {node.name: node.parameter_space() for node in self.__nodes}

    @property
    def input_type(self) -> str:
        """
        The data type of the input this pipeline can process.

        This is the input data type of the first node defined in this
        pipeline.
        This method will return the string "Any" if no node is added
        to the pipeline.

        :return: The data type of the input data
        """
        if not self.__nodes:
            return "Any"
        return self.__nodes[0].input_type

    @property
    def output_type(self) -> str:
        """
        The data type of the result this pipelines returns.

        This is the output data type of the last node defined in this
        pipeline.
        This method will return the string "Any" if no node is added
        to the pipeline.

        :return: The data type of the output data
        """
        if not self.__nodes:
            return "Any"
        return self.__nodes[-1].output_type

    def execute(self, input_data: Any, **kwargs: Dict[str, Any]) -> Any:
        """
        Execute the pipeline.

        By calling this, the whole pipeline is executed. Thus, the input data
        get passed to the first node in this pipeline and the results are
        passed along the pipeline. This method returns the last result of the
        pipeline.

        The `input_data` need to match the input type of the pipeline as returned
        by the `input_type` property.

        The `kwargs` are the arguments that should be used for the different nodes.
        These arguments need to match the names returned by the `parameter_space`
        method.

        The result of this method will match the output type of the last
        node in the pipeline as returned by the `output_type` property.

        :param input_data: The input data to execute the pipeline on.
        :param kwargs: The parameters to initialize the different nodes with.
        :return: The result of the pipeline
        """
        data = input_data
        for node in self.__nodes:
            parameters = kwargs.get(node.name, {})
            data = node.execute(data, **parameters)
        return data
