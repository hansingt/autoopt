#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
from utils import create_node


def test_name():
    try:
        _ = create_node().name
        raise AssertionError("No exception raised")
    except NotImplementedError:
        pass


def test_input_type():
    try:
        _ = create_node().input_type
        raise AssertionError("No exception raised")
    except NotImplementedError:
        pass


def test_output_type():
    try:
        _ = create_node().output_type
        raise AssertionError("No exception raised")
    except NotImplementedError:
        pass


def test_parameter_space():
    try:
        _ = create_node().parameter_space()
        raise AssertionError("No exception raised")
    except NotImplementedError:
        pass


def test_execute():
    try:
        _ = create_node().execute(None)
        raise AssertionError("No exception raised")
    except NotImplementedError:
        pass


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
