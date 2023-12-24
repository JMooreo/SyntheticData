import os
import random

import numpy as np

from clustering.BallTree import find_clusters_ball_tree
from datasets.Dataset import Dataset
from file_io.WriteCSV import write_to_csv
from sampling.Sampler import Sampler
from sampling.feature_selection.FeatureSelection import random_choose_feature
from weights.Weights import get_weights, WeightingStrategy


class SyntheticDataSampler(Sampler):

	def __init__(self, dataset: Dataset, max_radius: float, min_neighbors: int, max_neighbors: int,
				 weighting_strategy: WeightingStrategy, percent_random_noise=0):

		super().__init__(dataset, max_radius, min_neighbors, max_neighbors)
		self.weighting_strategy = weighting_strategy
		self.percent_random_noise = percent_random_noise
		self.weighting_strategy = weighting_strategy
		self.percent_random_noise = percent_random_noise

	def sample(self, targets=None, debug=False):
		np.random.seed(self.dataset.seed)
		random.seed(self.dataset.seed)
		self.dataset.load()

		targets = targets if targets is not None else self.dataset.embeddings

		cluster_indices, cluster_distances, source_entity_indices = find_clusters_ball_tree(
			targets, self.max_radius, self.min_neighbors, self.max_neighbors, self.dataset)

		cluster_meta = [self.dataset.metadata[ix] for ix in cluster_indices]

		if len(source_entity_indices) == 0 or len(cluster_meta) == 0:
			return np.array([]), np.array([]), np.array([])

		# Source entities are the entities that were used to find the cluster
		source_entities = self.dataset.metadata[source_entity_indices]
		entity_size = cluster_meta[0].shape[1]

		if self.dataset.create_id:
			entity_size += 1

		synthetic_data = np.zeros((len(cluster_meta), entity_size), dtype="O")

		for entity_id, (cluster, source_entity) in enumerate(zip(cluster_meta, source_entities)):
			distances = cluster_distances[entity_id]
			entity_weights = get_weights(self.weighting_strategy, distances, max_override=None)

			new_entity = np.zeros(entity_size, dtype="O")
			for i in range(cluster.shape[1]):
				if i == 0 and self.dataset.create_id:
					new_entity[i] = str(entity_id)

				column = cluster[:, i]
				column_type = self.dataset.column_types[i]
				precision = self.dataset.column_precision[i]
				bin_edges = self.dataset.bin_edges[i]
				categorical_distribution = self.dataset.categorical_feature_distributions[self.dataset.headers[i]]

				attribute_index = i

				if self.dataset.create_id:
					attribute_index += 1

				new_entity[attribute_index] = random_choose_feature(
					data=column,
					column_type=column_type,
					entity_weights=entity_weights,
					weighting_strategy=self.weighting_strategy,
					precision=precision, feature_bin_edges=bin_edges,
					source_feature=source_entity[i],
					feature_categorical_distribution=categorical_distribution,
					percent_random_noise=self.percent_random_noise
				)

			synthetic_data[entity_id] = new_entity

			if debug:
				print("Generated Entity:", entity_id)

		return synthetic_data

	def save(self, synthetic_data: np.ndarray, output_directory: str):
		try:
			os.mkdir(output_directory)
		except OSError:
			pass

		headers = self.dataset.get_headers()
		filepath = f"{output_directory}/" \
				   f"R_0_to_{round(self.max_radius, 3)}_" \
				   f"N_{round(self.min_neighbors)}_to_{round(self.max_neighbors)}_" \
				   f"{self.weighting_strategy}_" \
				   f"bin_res_{self.dataset.bin_resolution}_" \
				   f"noise_{round(self.percent_random_noise, 2)}_.csv"

		write_to_csv(synthetic_data, headers, filepath)
