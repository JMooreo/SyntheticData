import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from datasets.Dataset import Dataset
from machine_learning.Preprocessing import load_data_and_clean_columns


def get_train_tune_test(training_data: DataFrame, dataset: Dataset):
    x_train, x_test, y_train, y_test, _ = load_data_and_clean_columns(training_data, dataset)

    if len(x_train) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    x_train, x_tune, y_train, y_tune = train_test_split(
        x_train, y_train, test_size=0.15, random_state=30, stratify=y_train)

    return x_train, x_tune, x_test, y_train, y_tune, y_test
