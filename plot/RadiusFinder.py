import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from plot.Histogram import plot_histogram
from plot.Plot import Plot


class RadiusFinderPlot(Plot):

    def __init__(self, dataset, min_cluster_radii, max_cluster_radii, min_neighbors, max_neighbors, max_radius):
        self.dataset = dataset
        self.min_cluster_radii = min_cluster_radii
        self.max_cluster_radii = max_cluster_radii
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius

    def show(self):
        fig, (axis1, axis2, axis3) = plt.subplots(nrows=3, sharex=True)
        upper = np.maximum(np.max(self.min_cluster_radii), np.max(self.max_cluster_radii))
        bins = np.linspace(0, upper, 100)
        fig.set_size_inches(6, 2)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        plot_failure_rate_curve(self.dataset, self.min_cluster_radii, axis1, self.dataset.title, "black")
        plot_radii_histogram(self.min_cluster_radii, axis2,
                             f"Min Cluster Radius (min radius: 0, min neighbors: {self.min_neighbors})", "#e39a41",
                             bins)
        plot_radii_histogram(self.max_cluster_radii, axis3,
                             f"Max Cluster Radius (max radius: {self.max_radius}, max neighbors: {self.max_neighbors})",
                             "#e3747a", bins)

        axis3.set_xlim((0, upper))
        axis3.set_xlabel("Radius")

        plt.autoscale()
        plt.show()


def plot_radii_histogram(radii, axis, title, color, bins):
    if len(radii) != 0:
        plot_histogram(axis, radii, bins, title, color)
    else:
        axis.set_title("Couldn't create a histogram. No Data.")


def plot_failure_rate_curve(dataset, critical_radii, axis, title, color):
    cutoffs = np.linspace(0, np.nanmax(critical_radii), 1000)
    expected_length = len(dataset.embeddings)

    # Number of successes
    y = np.array([np.sum(critical_radii < cutoff) for cutoff in cutoffs])

    # Number of failures as a percentage
    y = (expected_length - y) / expected_length

    axis.plot(cutoffs, y*100, color=color)
    axis.grid(axis='y', alpha=0.75)
    axis.set_ylabel('Failure Rate')
    axis.set_xlim((cutoffs[0], cutoffs[-1]))
    axis.set_title(title)

    percentages = [1, 0.1, 0.05, 0.01, 0]
    colors = ["red", "darkorange", "orange", "green", "black"]

    return_cutoffs = []

    for percentage, color in zip(percentages, colors):
        if percentage == 0:
            cutoff = np.max(cutoffs)
        else:
            cutoff = cutoffs[np.argmax(y < percentage)]

        return_cutoffs.append(cutoff)
        axis.axvline(x=cutoff, color=color, label=f"{100*percentage}% @ {round(cutoff, 3)}")

    axis.yaxis.set_major_formatter(mtick.PercentFormatter())

    axis.legend()

    return return_cutoffs


def get_failure_rate_cutoffs(dataset, critical_radii, percentages=None):
    if percentages is None:
        percentages = [1, 0.1, 0.05, 0.01]

    cutoffs = np.linspace(0, np.nanmax(critical_radii), 1000)
    expected_length = len(dataset.embeddings)

    # Number of successes
    y = np.array([np.sum(critical_radii < cutoff) for cutoff in cutoffs])

    # Number of failures as a percentage
    y = (expected_length - y) / expected_length

    return_list = [cutoffs[np.argmax(y < percentage)] if percentage > 0 else np.max(cutoffs)
                   for percentage in percentages]
    return return_list[0] if len(return_list) == 1 else return_list
