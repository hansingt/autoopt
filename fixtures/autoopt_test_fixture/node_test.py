#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
from autoopt.distributions import QUniform, Normal, Choice


class NodeTest(object):
    def __init__(self, no_default, int_param=1, float_param=1.0, bool_param=True, list_param=None):
        self.no_default = no_default
        self.int_param = int_param
        self.float_param = float_param
        self.bool_param = bool_param
        self.list_param = list_param


@QUniform("int_param", min_value=0, max_value=10, q=1)
@Normal("float_param", loc=0, scale=1.0)
@Choice("bool_param", choices=[True, False])
@Choice("list_param", choices=["Apple", "Bananas"])
class NodeTestWithDecorator(NodeTest):
    pass
