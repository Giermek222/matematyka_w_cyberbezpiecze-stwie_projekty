from sympy import Matrix
from math import cos, sin, atan2

def jacobi_method(A, tol=1e-6, max_iterations=1000):
    n = A.shape[0]  
    eigenvectors = [Matrix.eye(n)] * n  

    for i in range(max_iterations):
        max_off_diag = 0
        p, q = None, None
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_off_diag:
                    max_off_diag = abs(A[i, j])
                    p, q = i, j

        # Check for convergence
        if max_off_diag < tol or i >= max_iterations:
            break

        phi = 0.5 * atan2(2 * A[p, q], A[q, q] - A[p, p])

        Q = Matrix.eye(n)
        Q[p, p] = cos(phi)
        Q[q, q] = cos(phi)
        Q[p, q] = -sin(phi)
        Q[q, p] = sin(phi)

        A = Q.T * A * Q
        eigenvectors = [eigvec * Q for eigvec in eigenvectors]


    # Extract eigenvalues from the diagonal of A
    return [[A[i, i] for i in range(n)], eigenvectors]
