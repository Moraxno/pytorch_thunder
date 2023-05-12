import numpy as np
from logging import getLogger

from pytorch_thunder.features import scotts_rule

log = getLogger(__file__)


def mutual_information_feature_select(
    x: np.ndarray,
    y: np.ndarray,
    num_features_or_beta: int,
    max_iter: int = 25,
    lower_beta: float = 0.0,
    upper_beta: float = 1.0,
):
    if num_features_or_beta < 0.0:
        raise ValueError()
    elif isinstance(num_features_or_beta, float) or num_features_or_beta == 0:
        return mutual_information_feature_select_by_beta(x, y, num_features_or_beta)
    else:
        return mutual_information_feature_select_by_num_features(
            x, y, num_features_or_beta, max_iter, lower_beta, upper_beta
        )


def mutual_information_feature_select_by_num_features(
    x: np.ndarray,
    y: np.ndarray,
    num_features: int,
    max_iter: int,
    lower_beta: float,
    upper_beta: float,
):
    assert num_features > 0
    assert max_iter > 0

    assert lower_beta <= upper_beta

    num_found_features = 0

    iter_cnt = 0
    while iter_cnt < max_iter:
        iter_cnt += 1

        new_beta = 0.5 * (lower_beta + upper_beta)
        log.info(f"Iteration {iter_cnt}/{max_iter}.")
        log.info(f"Guessing beta to be {new_beta}.")
        features = mutual_information_feature_select_by_beta(x, y, new_beta)
        num_found_features = len(features)

        log.info(f"Found {num_found_features} features and need {num_features}.")

        if num_found_features < num_features:
            upper_beta = new_beta
        elif num_found_features > num_features:
            lower_beta = new_beta
        else:
            return features, new_beta

    if iter_cnt >= max_iter:
        return features[:num_features], np.nan


def mutual_information_feature_select_by_beta(
    x: np.ndarray, y: np.ndarray, beta: float = 0.04
):
    assert beta >= 0.0
    assert beta <= 1.0

    num_features = x.shape[1]

    x_bin_edges_list = [make_clean_bins_from_data(x_col) for x_col in x.T]
    y_bin_edges = make_clean_bins_from_data(y)

    x_probs_list = [
        calculate_probabilities_with_nans(x_col, bin_edges)
        for x_col, bin_edges in zip(x.T, x_bin_edges_list)
    ]
    y_probs = calculate_probabilities_with_nans(y, y_bin_edges)

    I_xx = np.zeros((num_features, num_features))
    I_xy = np.zeros(num_features)

    for ix1, (x1_data, x1_bin_edges, x1_probs) in enumerate(
        zip(x.T, x_bin_edges_list, x_probs_list)
    ):
        # fill I_xx
        for ix2, (x2_data, x2_bin_edges, x2_probs) in enumerate(
            zip(x.T, x_bin_edges_list, x_probs_list)
        ):
            x1x2_probs = calculate_2d_probabilities_with_nans(
                x1_data, x2_data, x1_bin_edges, x2_bin_edges
            )
            mi = mutual_information(x1_probs, x2_probs, x1x2_probs)
            I_xx[ix1, ix2] = mi

        # fill I_y
        xy_probs = calculate_2d_probabilities_with_nans(
            x1_data, y, x1_bin_edges, y_bin_edges
        )
        mi = mutual_information(x1_probs, y_probs, xy_probs)
        I_xy[ix1] = mi

    features = joint_mutual_information_feature_select(I_xx, I_xy, beta)
    return features


def joint_mutual_information_feature_select(I_xx, I_xy, beta):
    selected_features = []
    num_features = len(I_xx)

    while len(selected_features) < num_features:
        best_feature_idx = I_xy.argmax()
        best_mi_to_target = I_xy[best_feature_idx]

        if best_mi_to_target < 0:
            break

        selected_features.append(best_feature_idx)

        selected_mi_to_other_features = I_xx[:, best_feature_idx]

        I_xy -= beta * selected_mi_to_other_features
        I_xy[best_feature_idx] = np.NINF

    return np.array(selected_features)


def calculate_probabilities_with_nans(data, bins):
    nan_mask = np.isnan(data)
    clean_data = data[~nan_mask]

    counts, _ = np.histogram(clean_data, bins)

    nan_counts = nan_mask.sum()

    master_counts = np.empty((counts.shape[0] + 1,), dtype=int)
    master_counts[:-1] = counts
    master_counts[-1] = nan_counts

    master_probs = master_counts / master_counts.sum()

    return master_probs


def numel(arr):
    return np.prod(arr.shape)


def calculate_2d_probabilities_with_nans(x_data, y_data, x_bins, y_bins):
    if numel(x_data) != numel(y_data):
        raise ValueError(f"x has {numel(x_data)} elements, but y has {numel(y_data)}!")

    if x_data.shape != y_data.shape:
        y_data = y_data.reshape(x_data.shape)

    x_nan_mask = np.isnan(x_data)
    y_nan_mask = np.isnan(y_data)

    both_nan_mask = x_nan_mask & y_nan_mask

    clean_x_data = x_data[~both_nan_mask]
    clean_y_data = y_data[~both_nan_mask]

    counts, _, _ = np.histogram2d(clean_x_data, clean_y_data, [x_bins, y_bins])

    x_counts_when_x_is_nan, _ = np.histogram(x_data[y_nan_mask], x_bins)
    y_counts_when_x_is_nan, _ = np.histogram(y_data[x_nan_mask], y_bins)

    both_nan_counts = both_nan_mask.sum()

    master_counts = np.empty((counts.shape[0] + 1, counts.shape[1] + 1), dtype=int)
    master_counts[:-1, :-1] = counts
    master_counts[-1, :-1] = y_counts_when_x_is_nan
    master_counts[:-1, -1] = x_counts_when_x_is_nan
    master_counts[-1, -1] = both_nan_counts

    master_probs = master_counts / master_counts.sum()

    return master_probs


def make_clean_bins_from_data(data):
    nan_mask = np.isnan(data)
    clean_data = data[~nan_mask]

    if len(clean_data) == 0:
        min_ = 0
        max_ = 0
        mean = 0
    else:
        min_ = clean_data.min()
        max_ = clean_data.max()
        mean = clean_data.mean()

    # scott's bin width rule
    bin_width = scotts_rule(clean_data)

    bin_edges = construct_bin_edges(mean, min_, max_, bin_width)
    return bin_edges


def construct_bin_edges(center, min_, max_, bin_width):
    assert bin_width > 0.0
    assert max_ >= min_
    assert max_ >= center
    assert min_ <= center

    num_down_steps = np.ceil((center - bin_width / 2 - min_) / bin_width)
    num_up_steps = np.ceil((max_ - (center + bin_width / 2)) / bin_width)

    low_end = center - bin_width / 2 - num_down_steps * bin_width
    high_end = center + bin_width / 2 + num_up_steps * bin_width

    num_edges = int(num_down_steps + num_up_steps + 2)
    bin_edges = np.linspace(low_end, high_end, num_edges)

    return bin_edges


def mutual_information(x_probs, y_probs, xy_probs, eps: float = 1e-10):
    independent_probs_matrix = np.outer(x_probs, y_probs)
    independent_probs_matrix[independent_probs_matrix == 0] = 1
    ratio_matrix = np.divide(xy_probs, independent_probs_matrix)
    ratio_matrix[ratio_matrix == 0] = eps
    log_matrix = np.log2(ratio_matrix)
    entropies = np.multiply(xy_probs, log_matrix)

    return entropies.sum()
