#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import random

from autoopt.distributions.base import Distribution
from autoopt.pipelines.node import PipelineNode
from autoopt_test_fixture.node_test import NodeTest, NodeTestWithDecorator


def create_node(name=None):
    if name is None:
        name = "".join([chr(random.randint(ord("A"), ord("Z"))) for _ in range(5)])
    return PipelineNode(name)


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


def test_node_class_not_found():
    node = create_node(name="not_existing_node")
    try:
        _ = node.node_class
        raise AssertionError("No Exception raised")
    except ValueError:
        pass


def test_node_class():
    node = create_node(name="NodeTest")
    assert node.node_class == NodeTest


def test_parameter():
    node = create_node(name="NodeTest")
    parameters = node.parameters
    assert set(parameters) == {"int_param", "float_param", "bool_param", "list_param"}


def test_optimization_parameters_no_decorator():
    node = create_node(name="NodeTest")
    parameters = node.optimization_parameters
    assert set(parameters) == {"int_param", "float_param", "bool_param", "list_param"}


def test_optimization_parameters_with_decorators():
    node = create_node(name="NodeTestWithDecorator")
    check = [parameter.name
             for parameter in getattr(NodeTestWithDecorator, Distribution.PARAMETER_CLASS_ATTRIBUTE).values()]
    parameters = node.optimization_parameters
    assert set(parameters) == set(check)


def test_parameter_space():
    node = create_node(name="NodeTest")
    parameter_space = node.parameter_space()
    assert set([parameter.name for parameter in parameter_space.values()]) == \
        {"int_param", "float_param", "bool_param", "list_param"}


def test_parameter_space_with_decorator():
    node = create_node(name="NodeTestWithDecorator")
    check = getattr(NodeTestWithDecorator, Distribution.PARAMETER_CLASS_ATTRIBUTE).values()
    parameter_space = node.parameter_space()
    assert set(parameter_space.values()) == set(check)
