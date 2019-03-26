import unittest
from autoopt.distributions.base import Distribution


class MyDistribution(Distribution):
    def mean(self):
        return super(MyDistribution, self).mean()

    def pdf(self, x: object):
        return super(MyDistribution, self).pdf(x)

    def plot(self):
        return super(MyDistribution, self).plot()


def call_method(name, *args):
    dist = MyDistribution()
    try:
        getattr(dist, name)(*args)
        raise AssertionError("No error raised")
    except NotImplementedError:
        pass


def test_mean():
    call_method("mean")


def test_pdf():
    call_method("pdf", 1)


def test_plot():
    call_method("plot")
