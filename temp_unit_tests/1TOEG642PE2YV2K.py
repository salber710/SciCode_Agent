
# Background: The weighted Jacobi method is an iterative algorithm used to solve linear systems of equations of the form Ax = b. 
# It is a variant of the Jacobi method that introduces a relaxation parameter, omega (ω), to potentially improve convergence. 
# The method involves decomposing the matrix A into its diagonal component D and the remainder R, such that A = D + R. 
# The weighted Jacobi iteration formula is derived from the equation Mx_{k+1} = Nx_k + b, where M = (1/ω)D and N = M - A. 
# The iteration step is then x_{k+1} = (1 - ω)x_k + ωD^{-1}(b - Rx_k). 
# The choice of ω is crucial for the convergence of the method, with ω = 2/3 often being optimal for minimizing the spectral radius of the iteration matrix. 
# The algorithm iterates until the change in the solution vector x between iterations is less than a specified tolerance, eps. 
# The residual is the L2 norm of the difference between Ax and b, and the error is the L2 norm of the difference between the current solution x and the true solution x_true.

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
    # Compute the inverse of the diagonal matrix D
    D_inv = np.diag(1 / np.diag(D))
    # Compute the remainder matrix R = A - D
    R = A - D
    
    # Initialize the current solution x with the initial guess x0
    x = x0
    # Initialize the previous solution x_prev with a copy of x0
    x_prev = np.copy(x0)
    
    # Iterate until the change in solution is less than the tolerance
    while True:
        # Compute the next iteration of x using the weighted Jacobi formula
        x = (1 - omega) * x_prev + omega * np.dot(D_inv, b - np.dot(R, x_prev))
        
        # Compute the L2 norm of the difference between the current and previous solution
        increment_norm = np.linalg.norm(x - x_prev, ord=2)
        
        # Check if the increment is less than the tolerance
        if increment_norm < eps:
            break
        
        # Update the previous solution
        x_prev = np.copy(x)
    
    # Compute the residual as the L2 norm of (Ax - b)
    residual = np.linalg.norm(np.dot(A, x) - b, ord=2)
    # Compute the error as the L2 norm of (x - x_true)
    error = np.linalg.norm(x - x_true, ord=2)
    
    return residual, error
