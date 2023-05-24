from typing import Optional, Sequence, Union

import numpy as np


def scotts_rule_statistics(num_data: int, std: Union[float, Sequence[float]]) -> float:
    std = np.array(std)

    if num_data <= 0:
        return np.zeros_like(std)

    return 3.49 * std / (num_data ** (1 / 3))


def scotts_rule_data(data: np.ndarray):
    if len(data.shape) > 2:
        raise ValueError("Expecting Array of dimensions (N, F) or (N, ).")

    std = np.std(data, axis=0)
    num_data = len(data)

    return scotts_rule_statistics(num_data, std)


def scotts_rule(
    num_data_or_data: Union[int, np.ndarray],
    std: Optional[Union[float, Sequence[float]]] = None,
):
    if isinstance(num_data_or_data, np.ndarray) and std is None:
        return scotts_rule_data(num_data_or_data)

    if isinstance(num_data_or_data, int) and std is not None:
        return scotts_rule_statistics(num_data_or_data, std)

    raise TypeError(
        "Either pass (np.ndarray, ) or (int, float) or (int, Sequence[float])."
    )
