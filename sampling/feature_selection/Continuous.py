import numpy as np
from sklearn.neighbors import KernelDensity

from weights.Weights import WeightingStrategy


def select_random_continuous_cluster_histogram(data, entity_weights, weighting_strategy, precision, bin_edges):
	if len(data) == 1:
		return data[0]

	data = data.astype(np.float32)
	data_max = np.nanmax(data)
	data_min = np.nanmin(data)

	if data_min == data_max:
		# Cant create any bins.
		return data[0]

	weighted_bin_counts, _ = np.histogram(data, bins=bin_edges, weights=entity_weights)
	bin_weights = weighted_bin_counts / np.nansum(weighted_bin_counts)

	if weighting_strategy == WeightingStrategy.TRUE_RANDOM:
		bin_weights = np.ones_like(bin_weights)

	# Select a bin according to the weight
	bin_indexes = np.arange(len(bin_edges) - 1)

	try:
		random_bin_index = np.random.choice(bin_indexes, p=bin_weights)
	except ValueError:
		print()

	bin_start = bin_edges[random_bin_index]
	bin_end = bin_edges[random_bin_index + 1]

	# Generate a random uniform number in that range
	return round(np.random.uniform(bin_start, bin_end), precision)


def select_random_continuous_cluster_based_kde(data, entity_weights, precision):
	# Visualize the Density
	weighted = data.astype(np.float32)
	nan_locs = np.isnan(weighted)
	sample = kde_sklearn(weighted[~nan_locs], entity_weights[~nan_locs])
	return round(sample, precision)


def kde_sklearn(x: np.ndarray, entity_weights):
	# find the best bandwidth
	# bandwidth_grid = GridSearchCV(KernelDensity(),
	# 							  {'bandwidth': np.linspace(0.01, 1, 30)},
	# 							  cv=5)
	# bandwidth_grid.fit(x[:, None])
	# bandwidth = bandwidth_grid.best_params_["bandwidth"]

	# Find the kernel density
	x_grid = np.linspace(np.min(x), np.max(x), 300)
	# kde_skl = KernelDensity(bandwidth=0.04)
	kde_skl = KernelDensity(bandwidth="silverman")
	kde_skl.fit(x[:, np.newaxis])
	log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
	density = np.exp(log_pdf)
	density /= np.sum(density)

	random_choice = np.random.choice(x_grid, size=1, p=density).item()

	# Visualize the choice
	# random_choice_index = np.argwhere(x_grid == random_choice).item()
	# hist, bin_edges = np.histogram(x, bins=20)
	# scaled_density = np.max(hist) * density / np.max(density)
	#
	# fig, (ax1, ax2) = plt.subplots(ncols=2)
	# ax1.plot(x_grid, scaled_density)
	# ax1.hist(x, bins=20)
	# ax1.scatter(x_grid[random_choice_index], scaled_density[random_choice_index], color="red")
	# ax2.hist(x, bins=20, weights=np.repeat(np.mean(entity_weights), len(x)))
	# ax2.hist(x, bins=20, weights=entity_weights)
	# plt.show()

	return random_choice
