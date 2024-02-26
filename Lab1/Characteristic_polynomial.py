from sympy import Matrix, eye, symbols, roots
import numpy as np

def characteristic_polynomial(matrix):
    x = symbols('x')
    eigenvectors = []

    identity = eye(matrix.shape[0], matrix.shape[1])
    
    output = matrix - identity * x
    result = roots(output.det())
    eigenvalues = [root for root, multiplicity in result.items() for _ in range(multiplicity)]

    for eigenvalue in result:
        A = matrix - eigenvalue * eye(matrix.shape[0])
        eigenvector = A.nullspace()
        for vector in eigenvector:
            eigenvectors.append(vector)

    return [eigenvalues, eigenvectors]
