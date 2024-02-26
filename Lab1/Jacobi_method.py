from sympy import Matrix, atan
import numpy as np
from math import cos, sin, atan2, floor

def jacobi_method(A, tol=1e-6, max_iterations=1000):
    n = A.shape[0]  
    eigenvectors = Matrix.eye(n)

    for i in range(max_iterations):

        max_off_diag = 0
        x, y = None, None
        for i in range(n):
            for j in range(n):
                if abs(A[i, j]) > max_off_diag and i != j:
                    max_off_diag = abs(A[i, j])
                    x,y = i, j

        # Check for convergence
        if max_off_diag < tol or i >= max_iterations:
            break

        phi = 0.5 * atan((2 * A[x, y]) / (A[y, y] - A[x, x]))

        Q = Matrix.eye(n)
        Q[x, x] = cos(phi)
        Q[y, y] = cos(phi)
        Q[x, y] = sin(phi)
        Q[y, x] = -sin(phi)

        A = Q.T * A * Q
        eigenvectors = eigenvectors * Q
    
    return [[A[i, i] for i in range(n)], eigenvectors]


matrix_B = Matrix([[-2,  0,  1],
                    [0, -2,  1],
                    [1,  1, -3]])

jacobi_method(matrix_B)
