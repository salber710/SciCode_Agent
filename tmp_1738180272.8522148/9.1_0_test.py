from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The weighted Jacobi method is an iterative algorithm for solving linear systems of the form Ax = b. 
# It is a variant of the Jacobi method that introduces a relaxation parameter, omega (ω), to potentially improve 
# convergence. The method decomposes the matrix A into its diagonal component D and the remainder R, such that A = D + R. 
# The weighted Jacobi iteration formula is x^(k+1) = (1 - ω)x^(k) + ωD^(-1)(b - Rx^(k)). The choice of ω is crucial for 
# the convergence rate, with ω = 2/3 often being optimal for certain matrices. The iteration continues until the change 
# in the solution vector is smaller than a specified tolerance, measured by the L2 norm. The residual is the L2 norm of 
# the difference between Ax and b, and the error is the L2 norm of the difference between the current solution and the 
# true solution x_true.


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
    # Compute the remainder matrix R = A - D
    R = A - D
    
    # Initialize the current solution with the initial guess
    x_k = x0
    # Initialize the previous solution to a vector of infinities to ensure the loop starts
    x_k_prev = np.inf * np.ones_like(x0)
    
    # Iterate until the change in solution is less than the tolerance
    while np.linalg.norm(x_k - x_k_prev, ord=2) >= eps:
        # Update the previous solution
        x_k_prev = x_k
        # Compute the next iteration of the solution
        x_k = (1 - omega) * x_k_prev + omega * np.dot(D_inv, b - np.dot(R, x_k_prev))
    
    # Compute the residual
    residual = np.linalg.norm(np.dot(A, x_k) - b, ord=2)
    # Compute the error with respect to the true solution
    error = np.linalg.norm(x_k - x_true, ord=2)
    
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