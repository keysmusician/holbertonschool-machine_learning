#!/usr/bin/env python3
""" Defines `determinant`. """


def omit(matrix, index):
    """ Omits a row and column from a square matrix. """
    side_length = len(matrix)
    column = index % side_length
    row = index // side_length

    decomp = list(matrix)
    del decomp[row]
    for i, row in enumerate(decomp):
        new_row = list(row)
        del new_row[column]
        decomp[i] = new_row

    return decomp


def determinant(matrix):
    """ Calculates the determinant of a matrix. """
    if not matrix or \
            type(matrix) is not list or \
            not all([type(element) is list for element in matrix]):
        raise TypeError('matrix must be a list of lists')

    if matrix == [[]]:
        return 1

    if not all([len(el) == len(matrix[0]) for el in matrix]) or \
            len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    else:
        SIGN = 1
        result = 0
        for i, n in enumerate(matrix[0]):
            result += n * determinant(omit(matrix, i)) * SIGN
            SIGN *= -1

        return result
