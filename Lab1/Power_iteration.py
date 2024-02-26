from sympy import Matrix, eye, symbols, roots

def power_iteration(matrix, vector, tol=1e-6, max_iterations=1000):
    vector /= vector.norm()

    for i in range(max_iterations):
        new_vector = matrix * vector  
        new_vector /= new_vector.norm()  
        
        # Check for convergence
        if (new_vector - vector).norm() < tol:
            break
        
        vector = new_vector  

    #round to tol
    eigenvalue = (vector.T * matrix * vector)[0,0] / (vector.T * vector)[0,0]
    eigenvalue = round(eigenvalue.evalf(), 5)
    
    eigenvector = [round(elem.evalf(), 5) for elem in vector] 

    return [eigenvalue, eigenvector]

