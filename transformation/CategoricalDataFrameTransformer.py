from typing import List

import numpy as np
from pandas import DataFrame


def convert_dataframes_to_categorical(df1: DataFrame, df2: DataFrame, continuous_columns: List[str], all_bin_edges: np.ndarray):
	""" Create a histogram and assign a bin index to each continuous value. """
	df1 = df1.copy()
	df2 = df2.copy()

	for column in continuous_columns:
		col_index = np.argwhere(df1.columns.values == column).item()
		bin_edges = all_bin_edges[col_index]
		df1_data = df1[column].values

		# For the source data, the top bin will ALWAYS only have one value.
		df1_bin_numbers = np.digitize(df1_data, bin_edges)
		df2_bin_numbers = np.digitize(df2[column].values, bin_edges)

		df1[column] = df1_bin_numbers
		df2[column] = df2_bin_numbers

	return df1, df2
