import numpy as np


def plot_histogram(axis, data, bin_edges, title, color):
	hist, _ = np.histogram(data, bins=bin_edges)
	bar_width = bin_edges[1] - bin_edges[0]
	axis.bar(bin_edges[:-1], hist, width=bar_width, edgecolor="black", linewidth=0.5, color=color,
			alpha=0.7, align="edge")
	axis.set_xlim(min(data) - bar_width, max(data) + bar_width)
	axis.grid(axis='y', alpha=0.75)
	axis.set_ylabel('Frequency')
	axis.set_title(title)
