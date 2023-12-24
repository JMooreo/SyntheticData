from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from datasets.ColumnType import ColumnType
from file_io.ReadCSV import read_csv_to_numpy_array


class Dataset:
	def __init__(self, title: str, dataset_directory: str, embeddings_path: str, metadata_path: str, column_types: List[ColumnType],
				 classification_label: str, column_precision: List[int], bin_resolution=25, remove_columns_before_processing: List[str] = None,
				 id_column_name="", create_synthetic_id=False, seed=42):
		self.title = title
		self.dataset_path = dataset_directory
		self.embeddings_path = embeddings_path
		self.metadata_path = metadata_path
		self.column_types = column_types
		self.classification_label = classification_label
		self.drop_labels = [] if remove_columns_before_processing is None else remove_columns_before_processing
		self.column_precision = column_precision
		self.bin_resolution = bin_resolution
		self.id_label = id_column_name
		self.create_id = create_synthetic_id
		self.seed = seed

		# Initialized by the Load function
		self.classification_column_index = 999999
		self.embeddings = np.array([])
		self.metadata = np.array([])
		self.headers = []
		self.bin_edges = []  # a list of bin edges for each continuous feature
		self.categorical_feature_distributions = {}

	def check(self):
		expected_length = len(self.headers)  # This is already after dropping the columns we don't care about.

		if any(length != expected_length for length in (len(self.column_precision), len(self.column_types), self.metadata.shape[1])):
			raise ValueError("Columns were not configured correctly or data was not loaded. All values should be the same."
							 f"\nHeaders: {len(self.headers)}"
							 f"\nColumn Precision: {len(self.column_precision)}"
							 f"\nColumn Types: {len(self.column_types)}"
							 f"\nMetadata: {self.metadata.shape[1]}")

	def get_headers(self):
		write_headers = list(filter(lambda header: header not in self.drop_labels, self.headers))
		return write_headers if not self.create_id else [self.id_label] + write_headers

	def load(self):
		self.embeddings, _ = read_csv_to_numpy_array(self.embeddings_path, drop_labels=["head"])
		self.metadata, self.headers = read_csv_to_numpy_array(self.metadata_path, self.drop_labels)
		self.classification_column_index = np.argwhere(self.headers == self.classification_label).item()
		self.check()
		self.init_categorical_and_continuous_distributions()

	def init_categorical_and_continuous_distributions(self):
		self.bin_edges = []
		self.categorical_feature_distributions = {}

		for i in range(self.metadata.shape[1]):
			if self.column_types[i] == ColumnType.CATEGORICAL:
				self.bin_edges.append([])
				self.categorical_feature_distributions[self.headers[i]] = Counter(self.metadata[:, i])
			elif self.column_types[i] == ColumnType.CONTINUOUS:
				self.categorical_feature_distributions[self.headers[i]] = Counter()
				data = self.metadata[:, i]

				data_min = np.nanmin(data) - 1e-6
				data_max = np.nanmax(data) + 1e-6

				num_bins = int(self.bin_resolution * (data_max - data_min) / np.nanstd(data))
				bin_edges = np.linspace(data_min, data_max, num_bins + 1)
				self.bin_edges.append(bin_edges)

	def load_dataframe_and_columns(self, path: str) -> Tuple[DataFrame, List[str], List[str]]:
		df = pd.read_csv(path, na_values=['', ' ']).drop(columns=self.drop_labels)

		categorical_columns = [col for col, col_type in zip(df.columns.values, self.column_types) if
							   col_type == ColumnType.CATEGORICAL]

		continuous_columns = [col for col, col_type in zip(df.columns.values, self.column_types) if
							  col_type == ColumnType.CONTINUOUS]

		return df, categorical_columns, continuous_columns
