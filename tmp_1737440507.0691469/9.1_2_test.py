import numpy as np



# Background: The weighted Jacobi method is an iterative algorithm used to solve the linear system Ax = b.
# In this method, the matrix A is decomposed into its diagonal component D and the remainder R such that A = D + R.
# The weighted Jacobi iteration formula is derived from the equation Mx = Nb, where M = (1/omega)D and N = M - A.
# The iterative update rule is x_k = (1 - omega)x_(k-1) + (omega * D^(-1) * (b - Rx_(k-1))).
# This method converges to the solution if A is strictly or irreducibly diagonally dominant, or if A is symmetric positive definite.
# The parameter omega is a relaxation parameter that can improve convergence speed, particularly when omega = 2/3.
# The stopping criterion is based on the difference between successive iterations: ||x_k - x_(k-1)||_2 < eps.
# The residual is calculated as ||Ax - b||_2 and the error is ||x - x_true||_2.


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
    
    # Initialize variables
    x_k = np.copy(x0)
    D = np.diag(np.diag(A))  # Extract diagonal matrix D
    R = A - D  # Compute remainder matrix R
    D_inv = np.linalg.inv(D)  # Inverse of the diagonal matrix D
    diff_norm = np.inf  # Initialize difference norm as infinity
    iter_count = 0  # Iteration counter

    while diff_norm >= eps:
        # Perform the weighted Jacobi iteration
        x_k_new = (1 - omega) * x_k + omega * (D_inv @ (b - R @ x_k))
        
        # Calculate the difference norm for stopping criterion
        diff_norm = np.linalg.norm(x_k_new - x_k, ord=2)
        
        # Update the current solution
        x_k = x_k_new
        iter_count += 1

    # Compute the residual and error
    residual = np.linalg.norm(A @ x_k - b, ord=2)
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
