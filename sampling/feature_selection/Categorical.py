import random
import numpy as np

from weights.Weights import WeightingStrategy


def select_random_categorical(column, entity_weights, weighting_strategy):
    column_freq = {}

    for feature, weight in zip(column, entity_weights):
        column_freq.setdefault(feature, 0)
        column_freq[feature] += weight

    choices = list(column_freq.keys())
    weights = list(column_freq.values())

    if weighting_strategy == WeightingStrategy.TRUE_RANDOM:
        weights = np.ones_like(weights)

    if len(weights) == 0:
        return np.array([])

    return random.choices(choices, weights=weights, k=1)[0]
