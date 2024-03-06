import numpy as np

def count_multiplicities(values):
    multiplicity_map = {}
    for value in values:
        if value in multiplicity_map:
            multiplicity_map[value] += 1
        else:
            multiplicity_map[value] = 1
    return multiplicity_map

def jordan_decomposition(matrix):
    eigenvalues, P = np.linalg.eig(matrix)
    J = np.zeros_like(matrix)
    eigenvalues_multiplicity = count_multiplicities(eigenvalues)

    position = 0
    for eigenvalue in eigenvalues:
        J[position, position] = round(eigenvalue,3)
        
        if (eigenvalues_multiplicity[eigenvalue] > 1):
            J[position, position+1] = 1
        eigenvalues_multiplicity[eigenvalue] -= 1
        position += 1

    return J, P
