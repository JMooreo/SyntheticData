import pkg_resources

from datasets.ColumnType import ColumnType
from datasets.Dataset import Dataset


def get_wine() -> Dataset:
	dataset_directory = wine_directory()
	column_types = [ColumnType.CATEGORICAL] + [ColumnType.CONTINUOUS for _ in range(11)] + [ColumnType.CATEGORICAL]

	return Dataset(
		title="Wine",
		dataset_directory=dataset_directory,
		embeddings_path=f"{dataset_directory}/wine_train_embeddings.csv",
		metadata_path=f"{dataset_directory}/wine_train.csv",
		classification_label="quality",
		column_precision=[6 for _ in range(len(column_types))],  # number of decimal places for continuous columns. ignored for categorical columns.
		column_types=column_types
	)


def wine_directory():
	return pkg_resources.resource_filename('datasets.wine', '')
