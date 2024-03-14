import numpy as np

def jacobi_method(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.float64(x0.copy()) 
    x_new = np.float64(np.zeros_like(x0))
    iterations = 0
    
    while iterations < max_iter:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if i != j:
                    sigma += np.dot(A[i, j], x_new[j]) 
            x_new[i] = (b[i] - sigma) / A[i, i]
            
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new.copy()
        iterations += 1
    return x_new


