import numpy as np

def diagonal_decomposition(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    D = np.diag(eigenvalues)
    return D, eigenvectors
