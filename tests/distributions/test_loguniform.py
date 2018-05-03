import unittest
import random
import numpy as np
from autoopt.distributions import LogUniform, QLogUniform
from autoopt.distributions.base import QMixin


class LogUniformTestCase(unittest.TestCase):

    def test_min(self):
        min_value = random.randint(1, 9)
        dist = LogUniform("test", min_value=min_value, max_value=10)
        self.assertEqual(min_value, dist.min_value)

    def test_max(self):
        max_value = random.randint(2, 10)
        dist = LogUniform("test", min_value=1, max_value=max_value)
        self.assertEqual(max_value, dist.max_value)

    def test_pdf(self):
        max_value = random.randint(2, 10)
        min_value = random.randint(1, max_value - 1)
        min_log = np.log(min_value)
        max_log = np.log(max_value)
        dist = LogUniform("test", min_value=min_value, max_value=max_value)
        x_space = np.linspace(min_value - 1, max_value + 1, num=1000)
        for x in x_space:
            y = dist.pdf(x)
            check = 1 / (x * (max_log - min_log)) if x >= 0 and min_value <= x <= max_value else 0
            self.assertEqual(check, y)

    def test_mean(self):
        max_value = random.randint(2, 10)
        min_value = random.randint(1, max_value - 1)
        dist = LogUniform("test", min_value=min_value, max_value=max_value)
        check = (max_value - min_value) / (np.log(max_value) - np.log(min_value))
        self.assertEqual(check, dist.mean())

    def test_invalid_interval(self):
        min_value = 10
        max_value = 1
        self.assertRaises(ValueError, LogUniform, "test", min_value=min_value, max_value=max_value)

    def test_negative_boundary(self):
        self.assertRaises(ValueError, LogUniform, "test", min_value=-1, max_value=1)

    def test_zero_boundary(self):
        self.assertRaises(ValueError, LogUniform, "test", min_value=0, max_value=1)

    def test_plot(self):
        max_value = random.randint(2, 10)
        min_value = random.randint(1, max_value - 1)
        dist = LogUniform("test", min_value=min_value, max_value=max_value)
        plot = dist.plot()
        try:
            from matplotlib import pyplot as plt
            self.assertIsInstance(plot, plt.figure().__class__)
        except ImportError:
            self.assertIsNone(plot)


class QLogUniformTestCase(unittest.TestCase):

    def test_pdf(self):
        max_value = random.randint(2, 10)
        min_value = random.randint(1, max_value - 1)
        min_log = np.log(min_value)
        max_log = np.log(max_value)
        q = 1
        dist = QLogUniform("test", min_value=min_value, max_value=max_value, q=1)
        x_space = np.linspace(min_value - 1, max_value + 1, num=1000)
        for x in x_space:
            y = dist.pdf(x)
            x_round = QMixin(q=q).round_to_q(x)
            check = 1 / (x_round * (max_log - min_log)) if x_round >= 0 and min_value <= x_round <= max_value else 0
            self.assertEqual(check, y)

    def test_mean(self):
        max_value = random.randint(2, 10)
        min_value = random.randint(1, max_value - 1)
        q = 1
        dist = QLogUniform("test", min_value=min_value, max_value=max_value, q=q)
        check = dist.round_to_q((max_value - min_value) / (np.log(max_value) - np.log(min_value)))
        self.assertEqual(check, dist.mean())

    def test_plot(self):
        max_value = random.randint(2, 10)
        min_value = random.randint(1, max_value - 1)
        try:
            import matplotlib
            matplotlib.use("AGG")
            from matplotlib import pyplot as plt
            plt.ioff()
            dist = QLogUniform("test", min_value=min_value, max_value=max_value, q=1)
            plot = dist.plot()
            self.assertIsInstance(plot, plt.figure().__class__)
        except ImportError:
            self.assertIsNone(plot)
