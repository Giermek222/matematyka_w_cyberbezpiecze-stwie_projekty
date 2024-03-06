import numpy as np


def lu_decomposition(matrix):
    row_size, column_size = matrix.shape
    if (row_size > column_size):
        return None, None
    L = np.eye(row_size)
    U = np.zeros((row_size, column_size))

    for row in range(row_size):
        for column in range(column_size):
            if row <= column:
                U[row][column] = matrix[row][column] - np.sum(L[row, :row] * U[:row, column])
            else:
                L[row][column] = (matrix[row][column] - np.sum(L[row, :column] * U[:column, column])) / U[column][column]
    return L, U