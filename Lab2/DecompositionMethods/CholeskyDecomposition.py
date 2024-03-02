import numpy as np

def cholesky_decomposition(A):
    n = len(A)
    L = np.zeros_like(A, dtype=float)
    
    for i in range(n):
        for j in range(i+1):
            if i == j:
                L[i, j] = np.sqrt(A[i, j] - np.sum(L[i, :j] ** 2))
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    
    return L
