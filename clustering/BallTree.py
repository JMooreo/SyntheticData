import numpy as np
from sklearn.neighbors import BallTree


def find_clusters_ball_tree(data, max_distance, min_neighbors, max_neighbors, dataset, min_distance=0):
	tree = BallTree(data)

	try:
		distances, indices = tree.query(data, k=max_neighbors)
	except:
		print()
	source_entity_indices = range(len(data))

	# Make sure we ignore the head embedding that we used to generate the cluster.
	# This will account for any random noise that we add, if any.
	for row in range(distances.shape[0]):
		distances[row][0] = 0.

	masked_indices = [
		indices[row, (distances[row] > 0) & (distances[row] >= min_distance) & (distances[row] <= max_distance)]
		for row in range(distances.shape[0])]
	filtered_indices = [ix for ix in masked_indices if len(ix) >= min_neighbors]

	masked_distances = [
		distances[row, (distances[row] > 0) & (distances[row] >= min_distance) & (distances[row] <= max_distance)]
		for row in range(distances.shape[0])]

	filtered_source_entity_indices = np.array([source_entity_indices[index] for index, ix in enumerate(masked_indices) if len(ix) >= min_neighbors])

	filtered_distances = [dist for dist in masked_distances if len(dist) >= min_neighbors]
	return filtered_indices, filtered_distances, filtered_source_entity_indices
