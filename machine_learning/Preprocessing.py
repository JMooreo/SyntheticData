import numpy as np
from pandas import DataFrame

from datasets.ColumnType import ColumnType
from datasets.Dataset import Dataset
import pandas as pd


def enforce_columns(df, column_names):
    data = np.zeros((len(df), len(column_names)), dtype=np.float32)

    for i, col in enumerate(column_names):
        if col in df.columns:
            data[:, i] = df[col]

    return DataFrame(data, columns=column_names)


def load_data_and_clean_columns(train_data: DataFrame, dataset: Dataset, test_data_path: str):
    test_data = pd.read_csv(test_data_path, na_values=['', ' '])
    combined = pd.concat((train_data, test_data))
    categorical_columns = [header for header, column_type in zip(dataset.headers, dataset.column_types)
                           if column_type == ColumnType.CATEGORICAL]

    classes = sorted(combined[dataset.classification_label].unique())

    categorical_columns_after_drop = [c for c in categorical_columns if c not in [dataset.classification_label, *dataset.drop_labels]]

    # Dummies will create a one-hot encoding for any categorical variables
    combined_column_names = pd.get_dummies(
        combined.drop([dataset.classification_label, *dataset.drop_labels], axis=1, errors="ignore"),
        columns=categorical_columns_after_drop
    ).columns.values

    # The classification label must be categorical, and only two options for Logistic Regression.
    y_train_dummies = pd.get_dummies(train_data[dataset.classification_label]).values
    y_test_dummies = pd.get_dummies(test_data[dataset.classification_label]).values

    if y_train_dummies.shape[1] == 1:
        # All the values are the same. We can't classify anything.
        return np.array([]), np.array([]), np.array([]), np.array([])

    x_train = pd.get_dummies(
        train_data.drop([dataset.classification_label, *dataset.drop_labels], axis=1, errors="ignore"),
        columns=categorical_columns_after_drop)

    x_test = pd.get_dummies(
        test_data.drop([dataset.classification_label, *dataset.drop_labels], axis=1, errors="ignore"),
        columns=categorical_columns_after_drop)

    x_train = enforce_columns(x_train, combined_column_names)
    x_test = enforce_columns(x_test, combined_column_names)

    if not np.array_equal(x_train.columns.values, x_test.columns.values):
        culprits = list(set([col for col in x_train.columns.values if col not in x_test.columns.values] + [col for col in x_test.columns.values if col not in x_train.columns.values]))
        raise ValueError(f"X Train/Test columns are misaligned: {culprits}")

    y_train = y_train_dummies
    y_test = y_test_dummies
    x_train = x_train.values
    x_test = x_test.values

    # Turns the multi-dimensional array into a 1-d array where the position of the 1 is the class.
    # e.g. [[0,1], [1,0], [1,0], [0,1] becomes [1, 0, 0, 1] where the number now represents the id of the class (e.g. Edible, Poisonous).
    if len(y_train) == 0:
        raise ValueError("Failed to run machine learning evaluation. Training data appears to be empty.\n"
                         "Double check to make sure there are no empty CSVs in the batch comparison folder.")

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    return x_train, x_test, y_train, y_test, classes
