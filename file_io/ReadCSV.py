import pandas as pd


def read_csv_to_numpy_array(file_path, drop_labels=None):
    drop_labels = [] if not drop_labels else drop_labels
    data = pd.read_csv(file_path, na_values=['', ' '])
    data = data.drop(drop_labels, axis=1, errors="ignore")
    return data.values, data.columns.values
