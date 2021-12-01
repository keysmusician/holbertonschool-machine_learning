#!/usr/bin/env python3
"""Defines `mat_mul`."""


def mat_mul(mat1, mat2):
    """Performs 2D-matrix multiplication."""
    if len(mat1) == 0 or len(mat1[0]) != len(mat2):
        return None
    # transpose matrix 2
    mat2 = list(zip(*mat2))
    dot_products = list()
    for mat1_row in mat1:
        dot_products.append([])
        for mat2_row in mat2:
            # Compute dot product
            products = [a * b for a, b in zip(mat1_row, mat2_row)]
            dot_product = sum(products)
            dot_products[-1].append(dot_product)

    return dot_products
