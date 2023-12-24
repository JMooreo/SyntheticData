import numpy as np
import scipy
from pandas import DataFrame


def cosine_similarity_stats_categorical(source_df: DataFrame, synthetic_df: DataFrame):
    if not np.array_equal(source_df.columns.values, synthetic_df.columns.values):
        raise ValueError("Columns are not the same!")

    cosine_similarities = np.zeros_like(source_df.columns.values, dtype=np.float32)

    for i, column in enumerate(source_df.columns.values):
        source_hist = source_df[column].value_counts().to_dict()
        synthetic_hist = synthetic_df[column].value_counts().to_dict()
        unique_keys = sorted(set(source_hist.keys()) | set(synthetic_hist.keys()))

        source_array = np.zeros_like(unique_keys, dtype=np.int32)
        synthetic_array = np.zeros_like(unique_keys, dtype=np.int32)

        for j, key in enumerate(unique_keys):
            source_array[j] = source_hist.get(key, 0)
            synthetic_array[j] = synthetic_hist.get(key, 0)

        source_array_normalized = source_array / np.sum(source_array)
        synthetic_array_normalized = synthetic_array / np.sum(synthetic_array)

        cosine_similarities[i] = 1 - scipy.spatial.distance.cosine(source_array_normalized, synthetic_array_normalized)

    return np.mean(cosine_similarities), np.std(cosine_similarities)
