import os

from datasets.Dataset import Dataset
from datasets.avila import get_avila
from datasets.mushroom import get_mushroom
from datasets.polish import get_polish
from datasets.wine import get_wine
from evaluation.ConfusionMatrixEvaluator import ConfusionMatrixEvaluator
from machine_learning.MachineLearningMethod import MachineLearningMethod
from plot.BatchComparisonPlot import BatchComparisonPlot
from plot.ConfusionMatrixPlot import ConfusionMatrixPlot
from plot.DendrogramPlot import DendrogramPlot
from plot.RadiusFinder import RadiusFinderPlot

from sampling.RadiusFinderSampler import RadiusFinderSampler
from sampling.SyntheticDataSampler import SyntheticDataSampler
from weights.Weights import WeightingStrategy


def create_synthetic_data_and_save_it(dataset: Dataset, output_directory):
	sampler = SyntheticDataSampler(
		dataset=dataset,
		min_neighbors=60,
		max_neighbors=360,
		max_radius=0.3,
		weighting_strategy=WeightingStrategy.LINEAR,
		percent_random_noise=0
	)

	synthetic_data = sampler.sample(debug=False)
	sampler.save(synthetic_data, output_directory=output_directory)


def create_the_radius_finder_plot(dataset: Dataset):
	sampler = RadiusFinderSampler(
		dataset=dataset,
		min_neighbors=10,
		max_neighbors=360,
		max_radius=0.25
	)

	min_cluster_radii, max_cluster_radii = sampler.sample()

	radius_finder_plot = RadiusFinderPlot(
		dataset=dataset,
		min_cluster_radii=min_cluster_radii,
		max_cluster_radii=max_cluster_radii,
		min_neighbors=sampler.min_neighbors,
		max_neighbors=sampler.max_neighbors,
		max_radius=sampler.max_radius
	)

	radius_finder_plot.show()


def create_a_confusion_matrix_from_synthetic_data(dataset: Dataset, synthetic_data_directory, test_data_path):

	for file in os.listdir(synthetic_data_directory):
		synthetic_data_path = f"{synthetic_data_directory}/{file}"

		confusion_matrix_evaluator = ConfusionMatrixEvaluator(
			dataset=dataset,
			machine_learning_method=MachineLearningMethod.XGBOOST,
			train_data_path=synthetic_data_path,
			test_data_path=test_data_path
		)

		matrix, classes, weighted_f1 = confusion_matrix_evaluator.evaluate()

		confusion_matrix_plot = ConfusionMatrixPlot(matrix, classes, title=file)
		confusion_matrix_plot.show()


def batch_compare_synthetic_data(dataset: Dataset, synthetic_data_directory, test_data_path):
	# All empty synthetic CSVs must be removed or they will cause this to fail.
	batch_comparison_plot = BatchComparisonPlot(
		dataset=dataset,
		synthetic_data_directory=synthetic_data_directory,
		test_data_path=test_data_path,
		duplicate_detection_bin_resolution=4  # Divide the distribution into bins based on standard deviation / bin_resolution
	)

	batch_comparison_plot.show()


def create_a_dendrogram(dataset: Dataset):
	plot = DendrogramPlot(
		dataset=dataset,
		depth=900,  # increase the depth to get more of the tree, but will take longer.
		cut_height=0.0125  # max distance before we call them clusters
	)
	plot.show()
	# plot.export_clusters(output_directory="C:/Users/Justi/Programming/CI491/datasets/Avila/synthetic_data")


def main():
	dataset = get_polish()
	synthetic_data_directory = dataset.dataset_path + "/synthetic_data"
	test_data_path = dataset.dataset_path + "/polish_test.csv"

	# create_a_dendrogram(dataset)
	# create_the_radius_finder_plot(dataset)
	# create_a_confusion_matrix_from_synthetic_data(dataset, synthetic_data_directory, test_data_path)
	# batch_compare_synthetic_data(dataset, synthetic_data_directory, test_data_path)
	# create_synthetic_data_and_save_it(dataset, synthetic_data_directory)


if __name__ == "__main__":
	main()
