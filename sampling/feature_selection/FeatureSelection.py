import numpy as np

from datasets.ColumnType import ColumnType
from sampling.feature_selection.Categorical import select_random_categorical
from sampling.feature_selection.Continuous import select_random_continuous_cluster_histogram


def decide_generate_missing_value(data):
    percent_missing = len(data[np.isnan(data)]) / len(data)
    return np.random.uniform(0, 1) < percent_missing


def random_choose_feature(data, column_type, entity_weights, weighting_strategy, precision, feature_bin_edges,
                          source_feature, feature_categorical_distribution, percent_random_noise):

    try:  # Accounts for 'isnan' failing when called on an object and not a float.
        if decide_generate_missing_value(data):
            return ""
    except TypeError:
        pass

    should_hide_true_data = np.random.uniform(0, 1, len(data)) < percent_random_noise

    if column_type == ColumnType.CATEGORICAL:
        if percent_random_noise > 0:
            data[should_hide_true_data] = np.random.choice(list(feature_categorical_distribution.keys()), size=np.count_nonzero(should_hide_true_data), replace=True)

        return select_random_categorical(data, entity_weights, weighting_strategy)

    if column_type == ColumnType.CONTINUOUS:
        if percent_random_noise > 0:
            data[should_hide_true_data] = np.random.uniform(low=feature_bin_edges[0], high=feature_bin_edges[-1], size=np.count_nonzero(should_hide_true_data))

        return select_random_continuous_cluster_histogram(data, entity_weights, weighting_strategy, precision, feature_bin_edges)

        # Alternative -> just target the same bin as the source entity but with random noise
        # bin_size = feature_bin_edges[1] - feature_bin_edges[0]
        # return source_feature + (2 * bin_size) * np.random.uniform() - bin_size

    raise ValueError("Invalid column type. Must be 'categorical' or 'continuous'")
