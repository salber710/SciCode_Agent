import numpy as np



# Background: 
# The weighted Jacobi method is an iterative algorithm used to solve systems of linear equations of the form Ax = b.
# It is a variation of the Jacobi method, where a weighting factor, ω, is introduced to potentially accelerate convergence.
# In this method, the matrix A is decomposed into its diagonal component D and the remainder R (A = D + R).
# The method iteratively updates the solution x using the formula: x^{(k+1)} = x^{(k)} + ωD^{-1}(b - Ax^{(k)}).
# The parameter ω (omega) is used to adjust the contribution of the correction term, and it is optimal when ω = 2/3 for many problems.
# The iterative process continues until the L2 norm of the difference between successive approximations is less than a specified tolerance.
# The residual is ||Ax - b||_2, and the error is ||x - x_true||_2, where x_true is the known true solution for reference.


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
    
    # Extract the diagonal elements of A
    D = np.diag(np.diag(A))
    # Compute the inverse of the diagonal matrix D
    D_inv = np.diag(1 / np.diag(D))
    # Calculate R as the remainder matrix A - D
    R = A - D
    # Initial guess
    x_k = x0
    # Iteration counter
    iteration = 0
    # Start the iterative process
    while True:
        # Calculate the next iteration
        x_k1 = x_k + omega * np.dot(D_inv, b - np.dot(A, x_k))
        # Compute the L2 norm of the increment
        increment_norm = np.linalg.norm(x_k1 - x_k, ord=2)
        # Update x_k
        x_k = x_k1
        # Check the stopping criterion
        if increment_norm < eps:
            break
        iteration += 1

    # Calculate the residual
    residual = np.linalg.norm(np.dot(A, x_k) - b, ord=2)
    # Calculate the error with respect to the true solution
    error = np.linalg.norm(x_k - x_true, ord=2)
    
    return residual, error

from scicode.parse.parse import process_hdf5_to_tuple
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
