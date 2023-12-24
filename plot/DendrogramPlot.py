import csv
import json

import numpy as np

from datasets.Dataset import Dataset
from plot.Plot import Plot
import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt


class DendrogramPlot(Plot):

	def __init__(self, dataset: Dataset, depth=30, cut_height=None):
		self.dataset = dataset
		self.depth = depth
		self.cut_height = cut_height
		self.dataset.load()

	def show(self):
		linkage = hc.linkage(self.dataset.embeddings, metric="euclidean")
		max_distance_between_clusters = max(linkage[:, 2])
		threshold = self.cut_height if self.cut_height is not None else 0.7 * max_distance_between_clusters  # Consistent with MATLAB

		hc.dendrogram(linkage, truncate_mode="level", p=self.depth, color_threshold=threshold)
		plt.axhline(threshold, c="black")
		plt.show()

	# TODO: This is only functional for CATEGORICAL
	def export_clusters(self, output_directory, minimum_cluster_size=0):
		linkage = hc.linkage(self.dataset.embeddings, metric="euclidean")
		max_distance_between_clusters = max(linkage[:, 2])
		threshold = 0.7 * max_distance_between_clusters  # Consistent with MATLAB
		optimal_cut = hc.cut_tree(linkage, height=threshold)

		with open(self.dataset.metadata_path) as metadata:
			reader = csv.reader(metadata, delimiter=",")
			headers = next(reader)
			clusters = {}

			for (row, cluster_info) in zip(reader, optimal_cut):
				cluster_id = cluster_info[0]
				cluster = clusters.setdefault(cluster_id, {})

				for index, (header, feature) in enumerate(zip(headers, row)):
					if index == 0:
						continue  # Skip the ID

					cluster.setdefault(header, {}).setdefault(feature, 0)
					clusters[cluster_id][header][feature] += 1

			with open(output_directory + "/clusters.json", "w") as output_cluster_file:
				json_array = []

				for key, val in clusters.items():
					length = np.count_nonzero(optimal_cut == key)
					if length >= minimum_cluster_size:
						json_array.append({"length": length, "cluster": val})

				json.dump(json_array, output_cluster_file)
