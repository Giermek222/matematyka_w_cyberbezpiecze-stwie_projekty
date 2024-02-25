from sympy import Matrix, eye, symbols, roots

def power_iteration(matrix, vector):
    num_iterations=1000
    tol=1e-7 
    vector /= vector.norm()

    for i in range(num_iterations):
        new_vector = matrix * vector  # Matrix-vector multiplication
        new_vector /= new_vector.norm()  
        
        # Check for convergence
        if (new_vector - vector).norm() < tol:
            break
        
        vector = new_vector  # Update the eigenvector approximation

    #round to tol
    eigenvalue = (vector.T * matrix * vector)[0,0] / (vector.T * vector)[0,0]
    eigenvalue = round(eigenvalue.evalf(), 6)
    
    eigenvector = [round(elem.evalf()) for elem in vector] 
    print(eigenvector)

    return [eigenvalue, eigenvector]

