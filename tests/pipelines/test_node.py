#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import random

from autoopt.pipelines.node import PipelineNode


class DummyNode(PipelineNode):
    def parameter_space(self):
        return super(DummyNode, self).parameter_space()

    def execute(self, input_data, **kwargs):
        return super(DummyNode, self).execute(input_data, **kwargs)


def create_node(name=None):
    if name is None:
        name = "".join([chr(random.randint(ord("A"), ord("Z"))) for _ in range(5)])
    return DummyNode(name)


def test_name():
    name = "dummy"
    node = create_node(name=name)
    assert node.name == name


def test_equal_and_hash():
    name = "test"
    node1 = create_node(name=name)
    node2 = create_node(name=name)
    node3 = create_node(name="other")
    assert node1 == node2
    assert hash(node1) == hash(node2)
    assert node1 != node3
    assert hash(node1) != hash(node3)
    assert node1 != name


def test_parameter_space():
    node = create_node()
    try:
        _ = node.parameter_space()
        raise AssertionError("No exception raised")
    except NotImplementedError:
        pass


def test_make_parameter_name():
    node = create_node()
    parameter = "test"
    # noinspection PyProtectedMember
    parameter_name = node._make_parameter_name(parameter)
    check = "{node.name!s}_{parameter}".format(node=node, parameter=parameter)
    assert check == parameter_name


def test_execute():
    node = create_node()
    try:
        _ = node.execute(None)
        raise AssertionError("No exception raised")
    except NotImplementedError:
        pass
