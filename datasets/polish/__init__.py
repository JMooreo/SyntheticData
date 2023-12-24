import numpy as np
import pkg_resources

from datasets.ColumnType import ColumnType
from datasets.Dataset import Dataset


def get_polish() -> Dataset:
	dataset_directory = polish_directory()
	column_types = [ColumnType.CONTINUOUS for _ in range(64)] + [ColumnType.CATEGORICAL]

	return Dataset(
		title="Polish",
		dataset_directory=dataset_directory,
		embeddings_path=f"{dataset_directory}/polish_train_embeddings.csv",
		metadata_path=f"{dataset_directory}/polish_train.csv",
		classification_label="LABEL",
		create_synthetic_id=True,
		id_column_name="Company",
		remove_columns_before_processing=["Company"],  # always have to remove the ID column if there is one.
		column_precision=[6 for _ in range(len(column_types))],
		# number of decimal places for continuous columns. ignored for categorical columns.
		column_types=column_types
	)


def polish_directory():
	return pkg_resources.resource_filename('datasets.polish', '')
