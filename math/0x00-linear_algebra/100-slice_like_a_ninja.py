#!/usr/bin/env python3
"""Defines `np_slice`."""
import numpy as np


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes."""
    slices = []

    for i in range(matrix.ndim):
        t = axes.get(i)
        if t is not None:
            slices.append(slice(*t))
        else:
            slices.append(slice(None, None, None))

    return matrix[tuple(slices)]


mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                 [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                 [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
print()
print(mat2)
