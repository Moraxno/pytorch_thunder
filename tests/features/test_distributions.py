import numpy as np
import pytest

from pytorch_thunder.features.distributions import (
    scotts_rule_data,
    scotts_rule_statistics,
)


@pytest.mark.parametrize("N,std", [(10, 1.0), (2, 0.5), (144, 2.3)])
def test_scott_rule_statistics_on_valid_data(N, std):
    assert scotts_rule_statistics(N, std) > 0.0


@pytest.mark.parametrize(
    "data",
    [np.linspace(0, 1, 10), np.sin(np.linspace(0, np.pi, 100)), np.exp(np.arange(10))],
)
def test_scott_rule_data_on_valid_1d_data(data):
    assert scotts_rule_data(data) > 0.0


@pytest.mark.parametrize(
    "data",
    [
        np.linspace([0, 0], [10, 20], 10),
        np.sin(np.linspace([0, 0], [10, 20], 20)),
        np.exp(np.linspace([0, 0], [2, 3], 10)),
    ],
)
def test_scott_rule_data_on_valid_2d_data(data):
    assert (scotts_rule_data(data) > 0.0).all()


@pytest.mark.parametrize(
    "data",
    [
        np.linspace(0, 1, 10**3).reshape(10, 10, 10),
        np.sin(np.linspace(0, 1, 10**3).reshape(10, 10, 10)),
        np.exp(np.linspace(0, 1, 10**3).reshape(10, 10, 10)),
    ],
)
def test_scott_rule_data_on_3d_data_raises(data):
    with pytest.raises(ValueError):
        scotts_rule_data(data)
