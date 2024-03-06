import numpy as np

def qr_decomposition(matrix):
    row_size, column_size = matrix.shape
    

    
    Q = np.eye(row_size, dtype=float)  
    matrix = matrix.astype(float)  
    R = matrix.copy() 
    
    for column_index in range(min(row_size, column_size)):
        x = R[column_index:, column_index]
        householder_vector = x.copy()
        householder_vector[0] += np.sign(x[0]) * np.linalg.norm(x)  
        householder_vector = householder_vector / np.linalg.norm(householder_vector)  
        R[column_index:, :] -= 2 * np.outer(householder_vector, np.dot(householder_vector, R[column_index:, :]))
        Q[:, column_index:] -= 2 * np.outer(Q[:, column_index:].dot(householder_vector), householder_vector)
    return Q, R



