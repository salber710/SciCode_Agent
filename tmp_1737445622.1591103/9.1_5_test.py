from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The weighted Jacobi method is an iterative algorithm for solving a system of linear equations of the form Ax = b. 
# It is an extension of the classical Jacobi method, incorporating a relaxation parameter ω, which can accelerate the convergence of the method.
# The method splits the matrix A into its diagonal component D and the remainder R, i.e., A = D + R. 
# The iteration formula for the weighted Jacobi method is x^(k+1) = x^(k) + ω * D^(-1) * (b - A * x^(k)).
# The convergence of the method is determined by the spectral radius of the iteration matrix I - ω * D^(-1) * A.
# The parameter ω is typically chosen between 0 and 2, with ω = 2/3 often being optimal for certain types of matrices.
# The iteration continues until the change between successive approximations is below a specified tolerance (eps).
# This function also computes the L2 norm of the residual (||Ax - b||_2) and the error with respect to a true solution (||x - x_true||_2).


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
    D = np.diag(np.diag(A))  # Diagonal matrix from A
    R = A - D                # Remainder of A
    x = x0
    iteration_diff = np.inf  # Initialize the difference between iterations
    
    while iteration_diff >= eps:
        x_new = x + omega * np.linalg.inv(D) @ (b - A @ x)
        iteration_diff = np.linalg.norm(x_new - x, ord=2)
        x = x_new
    
    residual = np.linalg.norm(A @ x - b, ord=2)
    error = np.linalg.norm(x - x_true, ord=2)
    
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

