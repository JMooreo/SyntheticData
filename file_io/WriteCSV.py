import csv
from typing import List

import numpy as np


def write_to_csv(data: np.ndarray, headers: List[str], filename: str):
    with open(filename, 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
