from abc import abstractmethod

from datasets.Dataset import Dataset


class Sampler:
	def __init__(self, dataset: Dataset, max_radius: float, min_neighbors: int, max_neighbors: int):
		self.dataset = dataset
		self.max_radius = max_radius
		self.min_neighbors = min_neighbors
		self.max_neighbors = max_neighbors

	@abstractmethod
	def sample(self, **kwargs):
		pass
