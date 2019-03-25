#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
from setuptools import setup


setup(
    name="autoopt-test-fixture",
    version="0.1.0",
    packages=["autoopt_test_fixture"],
    entry_points={
        "autoopt.nodes": [
            "NodeTest = autoopt_test_fixture.node_test:NodeTest",
            "NodeTestWithDecorator = autoopt_test_fixture.node_test:NodeTestWithDecorator",
        ]
    }
)
