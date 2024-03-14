import numpy as np

def richardson_method(A, b, x0, alpha, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.float64(x0.copy()) 
    iterations = 0
    
    while iterations < max_iter:
        r = b - np.dot(A, x)
        x += alpha * r
        
        # Check for convergence
        if np.linalg.norm(r) < tol:
            break
        
        iterations += 1
    
    return x

