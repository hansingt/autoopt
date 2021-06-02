#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
"""
This module implement the base class for optimization tasks.

It can be inherited from in addons to implement other optimization tasks.
"""


__all__ = ["BaseTask"]


class BaseTask:
    """
    The base class for optimization tasks.

    It can be used as base in addons to implement other optimization tasks.
    """

    def __init__(self, input_path, optimizer):
        self.__input_path = input_path
        self.__optimizer = optimizer

    @property
    def input_path(self):
        """
        The path of the input data which should be used for the optimization.
        """
        return self.__input_path

    @property
    def optimizer(self):
        """
        The optimizer which will be used to optimize the processing pipelines.
        """
        return self.__optimizer

    def __getstate__(self):
        return {"input_path": self.input_path, "optimizer": self.optimizer}

    def __setstate__(self, state):
        self.__input_path = state["input_path"]
        self.__optimizer = state["optimizer"]
