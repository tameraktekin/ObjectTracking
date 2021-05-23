import numpy as np


def calculate_centroid(rect) -> object:
    left, top, right, bottom = rect
    center_x = int((left + right) / 2)
    center_y = int((top + bottom) / 2)
    return np.array([center_x, center_y])
