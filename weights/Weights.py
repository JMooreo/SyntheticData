from enum import Enum

import numpy as np

from weights.Linear import linear_weights


class WeightingStrategy(Enum):
    EVEN = "even"
    LINEAR = "linear"
    TRUE_RANDOM = "true_random"


def get_weights(weighting_strategy, distances, max_override=None):
    if weighting_strategy == WeightingStrategy.LINEAR:
        return linear_weights(distances, max_override)
    elif weighting_strategy in [WeightingStrategy.EVEN, WeightingStrategy.TRUE_RANDOM]:
        return np.ones(len(distances))
    else:
        raise ValueError(f"Weighting Strategy not recognized: {weighting_strategy}")
