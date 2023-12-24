import numpy as np
import pkg_resources

from datasets.ColumnType import ColumnType
from datasets.Dataset import Dataset


def get_mushroom() -> Dataset:
	dataset_directory = mushroom_directory()
	column_types = [ColumnType.CATEGORICAL for _ in range(23)]

	return Dataset(
		title="Mushroom",
		dataset_directory=dataset_directory,
		embeddings_path=f"{dataset_directory}/mushroom_train_embeddings.csv",
		metadata_path=f"{dataset_directory}/mushroom_train.csv",
		classification_label="target-label",
		create_synthetic_id=True,
		id_column_name="MSID",
		remove_columns_before_processing=["MSID"],  # always have to remove the ID column if there is one.
		column_precision=[np.nan for _ in range(len(column_types))],  # number of decimal places for continuous columns. ignored for categorical columns.
		column_types=column_types
	)


def mushroom_directory():
	return pkg_resources.resource_filename('datasets.mushroom', '')
