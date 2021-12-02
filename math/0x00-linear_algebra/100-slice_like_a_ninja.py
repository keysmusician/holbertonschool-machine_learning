#!/usr/bin/env python3
"""Defines `np_slice`."""


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes."""
    sorted_axes_items = list(axes.items())
    sorted_axes_items.sort
    max_index = sorted_axes_items[-1][0]
    index = [Ellipsis for _ in range(max_index + 1)]
    for item in sorted_axes_items:
        index[item[0]] = slice(*item[1])
    return matrix[tuple(index)]
