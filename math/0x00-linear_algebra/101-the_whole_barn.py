"""Defines `add_matrices`."""
import numpy as np


def add_matrices(mat1, mat2):
    """Adds two matrices."""
    mat1, mat2 = np.array(mat1), np.array(mat2)
    if not mat1.shape == mat2.shape:
        return None
    return mat1 + mat2
