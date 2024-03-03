import numpy as np

def ludecomposition(matrix):
    n = matrix.shape[0]
    L = np.eye(n, dtype = int)
    U = np.zeros((matrix.shape[0], matrix.shape[1]), dtype= int)

    for i in range(n):
        for j in range(i,n):
            s=0
            for k in range(i):
                s += (L[i,k]*U[k,j])
            U[i,j]=matrix[i,j]-s

        for j in range(i,n):
            s=0
            for k in range(i):
                s += (L[j,k]*U[k,i])
            L[j,i]=((1/U[i,i])*(matrix[j,i]-s))

    return L,U
