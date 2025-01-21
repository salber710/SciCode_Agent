import numpy as np



# Background: 
# The Weighted Jacobi method is an iterative technique for solving the linear system Ax = b. 
# It is a variant of the classical Jacobi method that introduces a relaxation parameter omega (ω) to potentially accelerate convergence. 
# The method starts with an initial guess x_0 and iteratively refines the solution. 
# The iterative formula for the Weighted Jacobi method is given by:
# x^(k+1) = x^(k) + ω * D^(-1) * (b - Ax^(k)), where D is the diagonal part of A.
# The choice of ω can significantly affect the convergence speed. An optimal choice is often ω = 2/3.
# The iteration continues until the change in successive approximations is less than a given tolerance (eps),
# specifically, until ||x^(k+1) - x^(k)||_2 < eps.
# During the iteration, it is also important to compute the residual ||Ax - b||_2 and the error ||x - x_true||_2
# at each step with respect to the true solution x_true.


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
    D_inv = np.diag(1 / np.diag(A))
    
    # Initialize variables
    x_k = x0
    residual = np.linalg.norm(np.dot(A, x_k) - b, 2)
    error = np.linalg.norm(x_k - x_true, 2)
    
    # Iterative process
    iteration = 0
    while True:
        # Calculate next approximation
        x_k1 = x_k + omega * np.dot(D_inv, (b - np.dot(A, x_k)))
        
        # Calculate the increment's norm
        increment_norm = np.linalg.norm(x_k1 - x_k, 2)
        
        # Update x_k
        x_k = x_k1
        
        # Calculate residual and error
        residual = np.linalg.norm(np.dot(A, x_k) - b, 2)
        error = np.linalg.norm(x_k - x_true, 2)
        
        # Check convergence
        if increment_norm < eps:
            break
        
        iteration += 1
    
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
