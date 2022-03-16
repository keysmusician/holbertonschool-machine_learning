#!/usr/bin/env python3
""" Defines `inverse`. """


def omit(matrix, index):
    """ Omits a row and column from a square matrix. """
    side_length = len(matrix)
    if type(index) is tuple and len(index) == 2:
        row, column = index
    elif type(index) is int:
        row = index // side_length
        column = index % side_length

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
    if not matrix or \
            type(matrix) is not list or \
            not all([type(element) is list for element in matrix]):
        raise TypeError('matrix must be a list of lists')
    elif matrix == [[]] or \
            not all([len(el) == len(matrix[0]) for el in matrix]) or \
            len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a non-empty square matrix')

    result = []
    for row_number, row in enumerate(matrix):
        result.append([])
        for column_number, _ in enumerate(row):
            result[row_number].append(
                determinant(omit(matrix, (row_number, column_number)))
            )

    return result


def cofactor(matrix):
    """ Calculates the cofactor matrix of a matrix. """
    minor_M = minor(matrix)
    row_sign = 1
    for row_number, row in enumerate(minor_M):
        element_sign = row_sign
        for column_number, element in enumerate(row):
            minor_M[row_number][column_number] *= element_sign
            element_sign *= -1
        row_sign *= -1
    return minor_M


def adjugate(matrix):
    """ Calculates the adjugate matrix of a matrix. """
    result = [list() for _row in matrix]

    for row in cofactor(matrix):
        for column_number, element in enumerate(row):
            result[column_number].append(element)

    return result


def inverse(matrix):
    """
    Calculates the inverse of a matrix.

    Returns: The inverse of `matrix`, or None if `matrix` is singular.
    """
    determinant_ = determinant(matrix)
    if not determinant_:
        return

    result = []

    for row_number, row in enumerate(adjugate(matrix)):
        result.append([])
        for element in row:
            result[row_number].append(element / determinant_)

    return result
