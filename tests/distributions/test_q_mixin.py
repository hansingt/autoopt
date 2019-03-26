#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Torben Hansing
#
import random

from autoopt.distributions.base import QMixin


def test_q():
    q = random.random()
    mixin = QMixin(q=q)
    assert mixin.q == q


def test_round_to_q():
    q = random.random()
    mixin = QMixin(q=q)
    for _ in range(10):
        value = random.random()
        check = round(value / q) * q
        assert check == mixin.round_to_q(value)
