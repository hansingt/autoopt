#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#


__all__ = ["BaseTask"]


class BaseTask(object):

    def __init__(self, input_path, optimizer):
        self.__input_path = input_path
        self.__optimizer = optimizer

    @property
    def input_path(self):
        return self.__input_path

    @property
    def optimizer(self):
        return self.__optimizer

    def __getstate__(self):
        return {
            "input_path": self.input_path,
            "optimizer": self.optimizer
        }

    def __setstate__(self, state):
        self.__input_path = state["input_path"]
        self.__optimizer = state["optimizer"]
