import numpy as np


def linear_weights(distances, max_override=None):
    if len(distances) == 0:
        return []

    max_distance = max_override if max_override is not None else np.max(distances)
    return (max_distance - distances) / max_distance
