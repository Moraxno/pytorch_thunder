import pytest
import numpy as np

from pytorch_thunder.features.mifs import (
    mutual_information_feature_select,
    make_clean_bins_from_data,
    construct_bin_edges,
)


def noisify_a_channel(data: np.ndarray, channel: int, amplitude: float = 0.01):
    data[:, channel] += np.random.random(data[:, channel].shape) * amplitude
    return data


def make_signal(dims, length=100, noise_dim=-1):
    signal = np.linspace([0] * dims, list(range(1, dims + 1)), length)
    if noise_dim >= 0:
        noisify_a_channel(signal, noise_dim)
    return signal


@pytest.mark.parametrize(
    "X,y,num,solution",
    [
        (make_signal(2), make_signal(1), 1, [0]),
        (make_signal(2), make_signal(1), 2, [0, 1]),
        (make_signal(2, noise_dim=0), make_signal(1), 2, [1, 0]),
    ],
)
def test_mifs_by_num(X, y, num, solution):
    f, _ = mutual_information_feature_select(X, y, num)
    assert len(f) == num
    assert (f == solution).all()


@pytest.mark.parametrize("signal", [np.sin(np.linspace(0, 2 * np.pi, 101))])
def test_make_clean_bins_from_data(signal):
    bins = make_clean_bins_from_data(signal)

    assert len(bins) > 2

    assert bins[0] <= signal.min()
    assert bins[-1] >= signal.max()


@pytest.mark.parametrize(
    "low,high,center,width,solution",
    [
        (-1, 1, 0, 0.5, [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]),
        (0, 4, 3.7, 1, [-0.8, 0.2, 1.2, 2.2, 3.2, 4.2]),
    ],
)
def test_bin_constructor(low, high, center, width, solution):
    bins = construct_bin_edges(center, low, high, width)

    assert bins == pytest.approx(solution)
