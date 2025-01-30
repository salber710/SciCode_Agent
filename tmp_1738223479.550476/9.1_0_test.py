from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The weighted Jacobi iteration is an iterative method used to solve linear systems of equations of the form Ax = b. 
# It is a variant of the Jacobi method, which is optimized by introducing a relaxation parameter, ω. 
# In this method, the matrix A is decomposed into its diagonal component D and the remainder R such that A = D + R. 
# The weighted Jacobi method uses M = (1/ω)D as the preconditioner. The update equation for the iterative process is:
# x^(k+1) = (1 - ω)x^(k) + ωD^(-1)(b - Rx^(k))
# This formulation is aimed at improving the convergence properties of the standard Jacobi method by minimizing the spectral radius of the iteration matrix.
# The convergence is checked by ensuring the L2 norm of the difference between successive approximations is below a given tolerance, eps.
# Moreover, the residual and the error norms are calculated at each iteration to analyze the convergence of the solution.


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
    
    # Diagonal matrix D
    D = np.diag(np.diag(A))
    
    # Off-diagonal part R = A - D
    R = A - D
    
    # Invert the diagonal matrix D
    D_inv = np.linalg.inv(D)
    
    # Initial guess
    x_k = x0.copy()
    
    # Initialize residual and error
    residual = np.linalg.norm(A @ x_k - b)
    error = np.linalg.norm(x_k - x_true)
    
    while True:
        # Compute the next iteration
        x_next = (1 - omega) * x_k + omega * D_inv @ (b - R @ x_k)
        
        # Check for convergence
        if np.linalg.norm(x_next - x_k) < eps:
            break
        
        # Update the current solution
        x_k = x_next
    
    # Calculate the final residual and error
    residual = np.linalg.norm(A @ x_k - b)
    error = np.linalg.norm(x_k - x_true)
    
    return residual, error


try:
    targets = process_hdf5_to_tuple('9.1', 3)
    target = targets[0]
    n = 7
    h = 1/(n-1)
    # A is a tridiagonal matrix with 2/h on the diagonal and -1/h on the off-diagonal
    diagonal = [2/h for i in range(n)]
    diagonal_up = [-1/h for i in range(n-1)]
    diagonal_down = [-1/h for i in range(n-1)]
    A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
    A[:, 0] = 0
    A[0, :] = 0
    A[0, 0] = 1/h
    A[:, -1] = 0
    A[-1, :] = 0
    A[7-1, 7-1] = 1/h
    b = np.array([0.1,0.1,0.0,0.1,0.0,0.1,0.1])
    x_true = np.linalg.solve(A, b)
    eps = 10e-5
    x0 = np.zeros(n)
    assert np.allclose(WJ(A, b, eps, x_true, x0,2/3), target)

    target = targets[1]
    n = 7
    h = 1/(n-1)
    # A is a tridiagonal matrix with 2/h on the diagonal and -1/h on the off-diagonal
    diagonal = [2/h for i in range(n)]
    diagonal_up = [-0.5/h for i in range(n-2)]
    diagonal_down = [-0.5/h for i in range(n-2)]
    A = np.diag(diagonal) + np.diag(diagonal_up, 2) + np.diag(diagonal_down, -2)
    b = np.array([0.5,0.1,0.5,0.1,0.5,0.1,0.5])
    x_true = np.linalg.solve(A, b)
    eps = 10e-5
    x0 = np.zeros(n)
    assert np.allclose(WJ(A, b, eps, x_true, x0, 1), target)

    target = targets[2]
    n = 7
    h = 1/(n-1)
    # A is a tridiagonal matrix with 2/h on the diagonal and -1/h on the off-diagonal
    diagonal = [2/h for i in range(n)]
    diagonal_2up = [-0.5/h for i in range(n-2)]
    diagonal_2down = [-0.5/h for i in range(n-2)]
    diagonal_1up = [-0.3/h for i in range(n-1)]
    diagonal_1down = [-0.5/h for i in range(n-1)]
    A = np.diag(diagonal) + np.diag(diagonal_2up, 2) + np.diag(diagonal_2down, -2) + np.diag(diagonal_1up, 1) + np.diag(diagonal_1down, -1)
    b = np.array([0.5,0.1,0.5,0.1,-0.1,-0.5,-0.5])
    x_true = np.linalg.solve(A, b)
    eps = 10e-5
    x0 = np.zeros(n)
    assert np.allclose(WJ(A, b, eps, x_true, x0, 0.5), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e