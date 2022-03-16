#!/usr/bin/env python3
""" Defines `minor`. """


def omit(matrix, index):
    """ Omits a row and column from a square matrix. """
    if type(index) is tuple and len(index) == 2:
        row, column = index
    elif type(index) is int:
        row = index // side_length
        column = index % side_length
    side_length = len(matrix)

    decomp = list(matrix)
    del decomp[row]
    for i, row in enumerate(decomp):
        new_row = list(row)
        del new_row[column]
        decomp[i] = new_row

    if not decomp:
        decomp.append([])
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


def minor(matrix):
    """ Calculates the minor matrix of a matrix. """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for e in matrix:
        if type(e) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(e) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    result = []
    for row_number, row in enumerate(matrix):
        result.append([])
        for column_number in range(len(row)):
            result[row_number].append(
                determinant(omit(matrix, (row_number, column_number)))
            )

    return result
