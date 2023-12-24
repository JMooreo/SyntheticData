from typing import List
from plot.Plot import Plot

import matplotlib.pyplot as plt
import numpy as np


class ConfusionMatrixPlot(Plot):
	def __init__(self, matrix: np.ndarray, classes: List[str], title: str):
		self.matrix = matrix
		self.classes = classes
		self.title = title if title else ""

	def show(self):
		fig, ax = plt.subplots()
		num_classes = len(self.classes)
		verbose_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
		verbose_matrix[:self.matrix.shape[0], :self.matrix.shape[1]] = self.matrix

		im = ax.imshow(verbose_matrix, cmap='Blues')

		# Add colorbar
		ax.figure.colorbar(im, ax=ax)

		# Set ticks and labels
		ax.set_xticks(np.arange(verbose_matrix.shape[0]))
		ax.set_yticks(np.arange(verbose_matrix.shape[1]))
		ax.set_xticklabels(self.classes)
		ax.set_yticklabels(self.classes)
		ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

		# Rotate the tick labels and set their alignment
		plt.setp(ax.get_xticklabels(), ha="center")

		max_val = np.max(self.matrix)

		# Loop over data to create annotations
		for i in range(verbose_matrix.shape[0]):
			for j in range(verbose_matrix.shape[1]):
				value = verbose_matrix[i, j]
				color = "white" if value > max_val * 0.3 else "black"
				ax.text(j, i, value, ha="center", va="center", color=color)

		# Set title and axis labels
		ax.set_xlabel("Predicted label")
		ax.set_ylabel("True label")
		ax.set_title(self.title)

		# Show the plot
		plt.show()
