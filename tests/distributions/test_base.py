import pytest
from autoopt.distributions.base import Distribution, _get_matplotlib


class MyDistribution(Distribution):
    def mean(self):
        return super(MyDistribution, self).mean()

    def pdf(self, x: object):
        return super(MyDistribution, self).pdf(x)

    def _plot_min_value(self):
        return super()._plot_min_value()

    def _plot_max_value(self):
        return super()._plot_max_value()

    def _plot_label(self):
        return super()._plot_label()


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


def test_plot_min_value():
    call_method("_plot_min_value")


def test_plot_max_value():
    call_method("_plot_max_value")


def test_plot_label():
    call_method("_plot_label")


@pytest.mark.skipif(_get_matplotlib() is None, reason="matplotlib not installed")
def test_plot_matlib():
    call_method("plot")


@pytest.mark.skipif(_get_matplotlib() is not None, reason="matplotlib installed")
def test_plot_no_matlib():
    dist = MyDistribution()
    assert dist.plot() is None
