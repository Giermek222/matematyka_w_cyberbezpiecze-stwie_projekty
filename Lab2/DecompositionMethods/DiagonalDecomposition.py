import numpy as np

def diagonal_decomposition(matrix):
    eigenvalues, P = np.linalg.eig(matrix)
    D = np.diag(eigenvalues)
    return D, P

