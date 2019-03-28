#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
from autoopt.pipelines import PipelineNode


def create_node(name=None, input_type=None, output_type=None, space=None, with_execute=False):
    class Node(PipelineNode):
        @property
        def name(self):
            if name is None:
                return super(Node, self).name
            return name

        @property
        def input_type(self):
            if input_type is None:
                return super(Node, self).input_type
            return input_type

        @property
        def output_type(self):
            if output_type is None:
                return super(Node, self).output_type
            return output_type

        def parameter_space(self):
            if space is None:
                return super(Node, self).parameter_space()
            return space

        def execute(self, input_data, **kwargs):
            if not with_execute:
                return super(Node, self).execute(input_data, **kwargs)
            return input_data
    return Node()


