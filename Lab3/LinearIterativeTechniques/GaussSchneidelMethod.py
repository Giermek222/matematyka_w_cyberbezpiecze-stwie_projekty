import numpy as np

def gauss_seidel_method(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    iterations = 0
    x = np.float64(x0.copy()) 
    x_new = np.float64(np.zeros_like(x))
    # Iterative process
    while iterations < max_iter:
        
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sigma) / A[i, i]
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new.copy()
        iterations += 1
    
    return x

