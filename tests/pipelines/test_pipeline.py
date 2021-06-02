#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import unittest

from autoopt.distributions import Choice
from autoopt.pipelines import Pipeline
from utils import create_node


class TestEmptyPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()

    def test_parameter_space(self):
        self.assertDictEqual(self.pipeline.parameter_space(), {})

    def test_input_type(self):
        self.assertEqual(self.pipeline.input_type, "Any")

    def test_output_type(self):
        self.assertEqual(self.pipeline.output_type, "Any")

    def test_execute(self):
        data = object()
        self.assertEqual(self.pipeline.execute(data), data)


class TestPipelineWithNodes(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            create_node(
                name="one",
                input_type="in",
                output_type="one_out",
                space={"param": Choice(choices=[True, False])},
                with_execute=True,
            ),
            create_node(
                name="two",
                input_type="one_out",
                output_type="two_out",
                space={"param": Choice(choices=[True, False])},
                with_execute=True,
            ),
            create_node(
                name="three",
                input_type="two_out",
                output_type="out",
                space={"param": Choice(choices=[True, False])},
                with_execute=True,
            ),
        ]
        self.pipeline = Pipeline()
        for node in self.nodes:
            self.pipeline.add_node(node)

    def test_parameter_space(self):
        check = self.pipeline.parameter_space()
        for node in self.nodes:
            self.assertIn(node.name, check)
            self.assertDictEqual(check[node.name], node.parameter_space())

    def test_input_type(self):
        self.assertEqual(self.nodes[0].input_type, self.pipeline.input_type)

    def test_output_type(self):
        self.assertEqual(self.nodes[-1].output_type, self.pipeline.output_type)

    def test_add_not_matching_node(self):
        node = create_node(name="not_matching", input_type="not_matching")
        self.assertRaises(ValueError, self.pipeline.add_node, node)

    def test_execute(self):
        data = object()
        self.assertEqual(data, self.pipeline.execute(input_data=data))
