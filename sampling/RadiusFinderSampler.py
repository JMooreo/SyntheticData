import random

import numpy as np

from clustering.BallTree import find_clusters_ball_tree
from sampling.Sampler import Sampler


class RadiusFinderSampler(Sampler):
	def sample(self, targets=None):
		np.random.seed(self.dataset.seed)
		random.seed(self.dataset.seed)
		self.dataset.load()

		targets = targets if targets is not None else self.dataset.embeddings

		cluster_indices, cluster_distances, source_entity_indices = find_clusters_ball_tree(
			targets, self.max_radius, self.min_neighbors, self.max_neighbors, self.dataset)

		# Collect statistics about the radius needed to meet the minimum and maximum neighbors
		# If the cluster was limited by the radius, then these values will be equal to the min/max radius.
		max_cluster_radii = np.zeros(len(cluster_distances), dtype=np.float32)
		min_cluster_radii = np.zeros_like(max_cluster_radii)

		for entity_id in range(len(cluster_distances)):
			distances = cluster_distances[entity_id]
			max_cluster_radii[entity_id] = np.max(distances)

			# Note: This assumes that we're not using the minimum radius.
			if len(distances) >= self.min_neighbors:
				min_cluster_radii[entity_id] = distances[int(self.min_neighbors - 1)]
			else:
				min_cluster_radii[entity_id] = np.NAN

		return min_cluster_radii, max_cluster_radii
