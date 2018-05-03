import unittest
from autoopt.distributions.base import Distribution


class MyDistribution(Distribution):
    def plot(self):
        return super(MyDistribution, self).plot()


class DistributionTestCase(unittest.TestCase):
    def setUp(self):
        self.name = "test"
        self.dist = MyDistribution(parameter_name=self.name)

    def test_name(self):
        self.assertEqual(self.name, self.dist.name)

    def test_call(self):
        def fuu(test):
            pass
        self.dist(fuu)
        self.assertIn(Distribution.PARAMETER_CLASS_ATTRIBUTE, dir(fuu))
        self.assertIn(self.dist, getattr(fuu, Distribution.PARAMETER_CLASS_ATTRIBUTE).values())

    def test_replace_with_multiple_calls(self):
        def fuu(test):
            pass
        self.dist(fuu)
        dist2 = MyDistribution(parameter_name=self.name)
        dist2(fuu)
        parameters = getattr(fuu, Distribution.PARAMETER_CLASS_ATTRIBUTE)
        self.assertIsNot(parameters[self.name], self.dist)
        self.assertIs(parameters[self.name], dist2)

    def test_eq(self):
        dist2 = MyDistribution(parameter_name=self.name)
        self.assertEqual(self.dist, dist2)

    def test_ne(self):
        dist2 = MyDistribution(parameter_name=self.name + "fuu")
        self.assertNotEqual(self.dist, dist2)

    def test_eq_not_comparable(self):
        self.assertNotEqual("fuu", self.dist)

    def test_str(self):
        self.assertEqual(self.name, str(self.dist))

    def test_repr(self):
        check = "{cls}<{name}>".format(cls=MyDistribution.__name__, name=self.name)
        self.assertEqual(check, repr(self.dist))

    def test_plot(self):
        self.assertRaises(NotImplementedError, self.dist.plot)
