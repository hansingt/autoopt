import unittest
import random
import numpy as np
from autoopt.distributions import LogUniform, QLogUniform


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

    @unittest.expectedFailure
    def test_invalid_interval(self):
        min_value = 10
        max_value = 1
        LogUniform("test", min_value=min_value, max_value=max_value)

    @unittest.expectedFailure
    def test_negative_boundary(self):
        LogUniform("test", min_value=-1, max_value=1)

    @unittest.expectedFailure
    def test_zero_boundary(self):
        LogUniform("test", min_value=0, max_value=1)


class QLogUniformTestCase(unittest.TestCase):

    def test_mean(self):
        max_value = random.randint(2, 10)
        min_value = random.randint(1, max_value - 1)
        q = 1
        dist = QLogUniform("test", min_value=min_value, max_value=max_value, q=q)
        check = dist.round_to_q((max_value - min_value) / (np.log(max_value) - np.log(min_value)))
        self.assertEqual(check, dist.mean())


if __name__ == '__main__':
    unittest.main()
