import unittest
import random
import numpy as np
from autoopt.distributions import Uniform, QUniform
from autoopt.distributions.base import QMixin


class UniformTestCase(unittest.TestCase):

    def test_min(self):
        min_value = random.randint(0, 9)
        dist = Uniform("test", min_value=min_value, max_value=10)
        self.assertEqual(min_value, dist.min_value)

    def test_max(self):
        max_value = random.randint(1, 10)
        dist = Uniform("test", min_value=0, max_value=max_value)
        self.assertEqual(max_value, dist.max_value)

    def test_pdf(self):
        max_value = random.randint(1, 10)
        min_value = random.randint(0, max_value - 1)
        dist = Uniform("test", min_value=min_value, max_value=max_value)
        x_space = np.linspace(min_value - 1, max_value + 1, num=1000)
        for x in x_space:
            y = dist.pdf(x)
            check = 1 / (max_value - min_value) if min_value <= x <= max_value else 0
            self.assertEqual(check, y)

    def test_mean(self):
        max_value = random.randint(1, 10)
        min_value = random.randint(0, max_value - 1)
        dist = Uniform("test", min_value=min_value, max_value=max_value)
        self.assertEqual(.5 * (min_value + max_value), dist.mean())

    @unittest.expectedFailure
    def test_invalid_interval(self):
        min_value = 10
        max_value = 1
        Uniform("test", min_value=min_value, max_value=max_value)


class QUniformTestCase(unittest.TestCase):

    def test_mean(self):
        max_value = random.randint(1, 10)
        min_value = random.randint(0, max_value - 1)
        q = 1
        dist = QUniform("test", min_value=min_value, max_value=max_value, q=q)
        round_to_q = QMixin(q=q).round_to_q
        self.assertEqual(round_to_q(.5 * (min_value + max_value)), dist.mean())


if __name__ == '__main__':
    unittest.main()
