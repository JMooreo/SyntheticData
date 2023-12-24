import pkg_resources

from datasets.ColumnType import ColumnType
from datasets.Dataset import Dataset


def get_avila() -> Dataset:
	dataset_directory = avila_directory()
	column_types = [ColumnType.CONTINUOUS for _ in range(10)] + [ColumnType.CATEGORICAL]

	return Dataset(
		title="Avila",
		dataset_directory=dataset_directory,
		embeddings_path=f"{dataset_directory}/avila_train_embeddings.csv",
		metadata_path=f"{dataset_directory}/avila_train.csv",
		classification_label="LABEL",
		create_synthetic_id=True,  # just sequential numbers. Needed if the dataset has an id column.
		id_column_name="PA",
		remove_columns_before_processing=["PA"],  # always have to remove the ID column if there is one.
		column_precision=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0],  # number of decimal places for continuous columns. ignored for categorical columns.
		column_types=column_types
	)


def avila_directory():
	return pkg_resources.resource_filename('datasets.avila', '')
