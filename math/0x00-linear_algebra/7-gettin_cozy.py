#!/usr/bin/env python3
"""Defines `cat_matrices2D`."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis."""
    mat1 = [row.copy() for row in mat1]
    mat2 = [row.copy() for row in mat2]
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        return [row + mat2[index] for index, row in enumerate(mat1)]



mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
print(mat4)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print(mat5)

# Mutate mat1:
mat1[0] = [9, 10]
mat1[1].append(5)
print(mat1)

# Ensure mutation does not affect return value of cat_matrices2D:
print(mat4)
print(mat5)
