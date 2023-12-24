import os

import numpy as np
import shutil

from datasets.Dataset import Dataset
from evaluation.ConfusionMatrixEvaluator import ConfusionMatrixEvaluator
from evaluation.Duplicates import find_duplicates_in_dataframes
from evaluation.Similarity import cosine_similarity_stats_categorical
from machine_learning.MachineLearningMethod import MachineLearningMethod
from plot.Plot import Plot
from transformation.CategoricalDataFrameTransformer import convert_dataframes_to_categorical
import matplotlib.pyplot as plt


class BatchComparisonPlot(Plot):
	def __init__(self, dataset: Dataset, synthetic_data_directory: str, test_data_path: str, duplicate_detection_bin_resolution: int):
		self.dataset = dataset
		self.synthetic_data_directory = synthetic_data_directory
		self.test_data_path = test_data_path
		self.dataset.load()

		# Use the dataset to create the ranges for detecting duplicates
		temp_bin_edges = self.dataset.bin_edges
		temp_bin_resolution = self.dataset.bin_resolution

		self.dataset.bin_resolution = duplicate_detection_bin_resolution
		self.dataset.init_categorical_and_continuous_distributions()

		self.duplicate_detection_bin_edges = self.dataset.bin_edges
		self.dataset.bin_edges = temp_bin_edges
		self.dataset.bin_resolution = temp_bin_resolution

	def show(self):
		source_data_file_name = self.dataset.metadata_path.split("/")[-1]

		# Copies the training data into the same folder as the synthetic data.
		# This is leftover from when we wanted to see the source data right next to the synthetic data
		# But they are in two separate graphs now.
		try:
			shutil.copyfile(self.dataset.metadata_path, f"{self.synthetic_data_directory}/{source_data_file_name}")
		except:
			pass

		source_dataframe, categorical_columns, continuous_columns = self.dataset.load_dataframe_and_columns(
			self.dataset.metadata_path)

		graph_data = []

		for file in os.listdir(self.synthetic_data_directory):
			synthetic_dataframe, _, _ = self.dataset.load_dataframe_and_columns(f"{self.synthetic_data_directory}/{file}")
			source_clean, synthetic_clean = convert_dataframes_to_categorical(
				source_dataframe, synthetic_dataframe, continuous_columns, self.duplicate_detection_bin_edges)

			(
				number_of_internal_source_duplicates,
				percent_internal_source_duplicates,
				number_of_internal_synthetic_duplicates,
				percent_internal_synthetic_duplicates,
				number_of_entities_replicated_from_source,
				percent_entities_replicated_from_source
			) = find_duplicates_in_dataframes(source_clean, synthetic_clean)

			mean_cos_similarity, standard_deviation_cos_similarity = cosine_similarity_stats_categorical(source_clean, synthetic_clean)
			success_rate = len(synthetic_dataframe) / len(source_dataframe)

			confusion_matrix_evaluator = ConfusionMatrixEvaluator(
				dataset=self.dataset,
				machine_learning_method=MachineLearningMethod.XGBOOST,
				train_data_path=f"{self.synthetic_data_directory}/{file}",
				test_data_path=self.test_data_path
			)

			confusion_matrix, classes, weighted_f1 = confusion_matrix_evaluator.evaluate()

			graph_data.append(
				{
					"Plot Data": {
						"Internal Duplicate %": {
							"value": 100 * percent_internal_synthetic_duplicates,
							"color": "red",
							"min": 0,
							"max": 100
						},
						"Source Replication %": {
							"value": 100 * percent_entities_replicated_from_source,
							"color": "red",
							"min": 0,
							"max": 100
						},
						"Similarity %": {
							"value": 100 * mean_cos_similarity,
							"color": "blue",
							"min": 0,
							"max": 100
						},
						"Weighted F1 Score": {
							"value": weighted_f1,
							"color": "teal",
							"min": 0,
							"max": 1,
						},
						"Success Rate": {
							"value": success_rate,
							"color": "lime"
						},
					},
					"Filename": file,
				}
			)

		source_graph_config = find_graph_config_and_remove(source_data_file_name, graph_data)
		graph_data = sorted(graph_data, key=lambda stats: stats["Plot Data"]["Weighted F1 Score"]["value"])

		plot_source_data_info(source_graph_config)
		plot_synthetic_data_info(graph_data)

		# TODO: This copying and removing really isn't necessary.
		try:
			os.remove(f"{self.synthetic_data_directory}/{source_data_file_name}")
		except:
			pass

		plt.tight_layout()
		plt.show()


def find_graph_config_and_remove(target: str, lst):
	found = None

	for obj in lst:
		if obj["Filename"] == target:
			found = obj
			break

	if found is not None:
		lst.remove(found)
		return found
	else:
		return None


def plot_source_data_info(source_graph_config):
	source_plot_data_keys = ["Internal Duplicate %", "Weighted F1 Score"]

	# Plot the Source Data
	source_fig, source_axes = plt.subplots(ncols=len(source_plot_data_keys))
	source_fig.set_size_inches(13, 1)

	try:
		len(source_axes)
	except TypeError:
		source_axes = [source_axes]

	for i, (axis, axis_label) in enumerate(zip(source_axes, source_plot_data_keys)):
		x = [0]
		axis_configs = [source_graph_config["Plot Data"][axis_label]]
		labels = [source_graph_config["Filename"]]

		plot_bar(axis, axis_label, axis_configs, x, labels, show_ticks=i == 0)


def set_axis_limits_and_ticks(axis, data, minimum=0, maximum=1):
	# data_min = min(data)
	# data_max = max(data)
	# data_range = data_max - data_min

	# a way to see the differences more clearly
	# if data_range == 0:
	# 	data_range = 1e-3
	#
	# true_min = round(max(minimum, data_min - data_range / 4), 2)
	# true_max = round(min(maximum, data_max + data_range / 4), 2)
	#
	# if true_min == maximum:
	# 	true_min -= 0.01
	#
	# if true_max == minimum:
	# 	true_max += 0.01
	#
	# if true_min == true_max:
	# 	true_min -= 0.01
	# 	true_max += 0.01

	true_min = minimum
	true_max = maximum

	raw_ticks = np.linspace(true_min, true_max, 3)
	axis.set_xlim((true_min, true_max))
	axis.set_xticks([round(v, 2) for v in raw_ticks])


def plot_bar(axis, axis_label, axis_configs, x, x_labels, show_ticks=False):
	axis_data = [c["value"] for c in axis_configs]
	bar_color = axis_configs[0]["color"]
	minimum = axis_configs[0].get("min", 0)
	maximum = axis_configs[0].get("max", 1)

	axis.barh(x, axis_data, align="center", alpha=0.5, color=bar_color)
	set_axis_limits_and_ticks(axis, axis_data, minimum, maximum)
	axis.set_xlabel(axis_label)
	axis.xaxis.grid(True)

	for x_val, y_val in zip(x, axis_data):
		axis.text(y_val, x_val, str(round(y_val, 2)), color='black', va='center')

	if show_ticks:
		axis.set_yticks(x)
		axis.set_yticklabels(x_labels)
	else:
		axis.set_yticks([])
		axis.set_yticklabels([])


def plot_synthetic_data_info(graph_data):

	if len(graph_data) == 0:
		print("There was no synthetic data to compare. Generate some first, or make sure the synthetic data directory is not empty.")
		return

	plot_data_keys = list(graph_data[0]["Plot Data"].keys())

	fig, axes = plt.subplots(ncols=len(plot_data_keys))
	fig.set_size_inches(13, 8)

	x = range(len(graph_data))
	labels = [d["Filename"] for d in graph_data]
	for i, (axis, axis_label) in enumerate(zip(axes, plot_data_keys)):
		axis_configs = [d["Plot Data"][axis_label] for d in graph_data]
		plot_bar(axis, axis_label, axis_configs, x, labels, show_ticks=i == 0)

