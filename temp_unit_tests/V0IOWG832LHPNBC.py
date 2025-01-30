
# Background: The weighted Jacobi method is an iterative algorithm used to solve systems of linear equations of the form Ax = b. 
# It is a variant of the Jacobi method that introduces a relaxation parameter, omega (ω), to potentially improve convergence. 
# The method involves decomposing the matrix A into its diagonal component D and the remainder R, such that A = D + R. 
# The weighted Jacobi iteration formula is derived from the equation Mx = (M - A)x + b, where M = (1/ω)D. 
# The iteration step is given by x^(k+1) = (1 - ω)x^(k) + ωD^(-1)(b - Rx^(k)). 
# The choice of ω is crucial for the convergence of the method, with ω = 2/3 often being optimal for minimizing the spectral radius of the iteration matrix. 
# The algorithm iterates until the change in the solution vector x is smaller than a specified tolerance, indicating convergence. 
# The residual is the L2 norm of the difference between Ax and b, and the error is the L2 norm of the difference between the current solution and the true solution.

import numpy as np

def WJ(A, b, eps, x_true, x0, omega):
    '''Solve a given linear system Ax=b with weighted Jacobi iteration method
    Input
    A:      N by N matrix, 2D array
    b:      N by 1 right hand side vector, 1D array
    eps:    Float number indicating error tolerance
    x_true: N by 1 true solution vector, 1D array
    x0:     N by 1 zero vector, 1D array
    omega:  float number shows weight parameter
    Output
    residuals: Float number shows L2 norm of residual (||Ax - b||_2)
    errors:    Float number shows L2 norm of error vector (||x-x_true||_2)
    '''
    
    # Extract the diagonal of A
    D = np.diag(np.diag(A))
    # Compute the inverse of the diagonal matrix
    D_inv = np.diag(1 / np.diag(D))
    # Compute the remainder matrix R
    R = A - D
    
    # Initialize the current solution
    x_k = x0
    # Initialize the previous solution
    x_k_prev = x0
    
    # Iterative process
    while True:
        # Compute the next iteration
        x_k = (1 - omega) * x_k_prev + omega * np.dot(D_inv, b - np.dot(R, x_k_prev))
        
        # Check for convergence
        if np.linalg.norm(x_k - x_k_prev, ord=2) < eps:
            break
        
        # Update the previous solution
        x_k_prev = x_k
    
    # Compute the residual
    residual = np.linalg.norm(np.dot(A, x_k) - b, ord=2)
    # Compute the error
    error = np.linalg.norm(x_k - x_true, ord=2)
    
    return residual, error
