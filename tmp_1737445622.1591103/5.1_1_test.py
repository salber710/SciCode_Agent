from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The Lanczos Iteration is an algorithm used to approximate eigenvalues and eigenvectors of a symmetric matrix.
# It is an iterative method that builds an orthonormal basis for the Krylov subspace. The input vector `b` is typically 
# normalized to start the process. The iteration constructs a tridiagonal matrix `T` whose eigenvalues approximate those of `A`.
# The columns of the matrix `Q` form an orthonormal basis, which is used to generate the tridiagonal form.
# The algorithm involves three main steps at each iteration: 
# 1. Compute the next vector in the Krylov sequence.
# 2. Orthogonalize this vector against all previous vectors in `Q`.
# 3. Normalize the resulting vector to maintain orthonormality.


def lanczos(A, b, m):
    '''Inputs:
    A : Matrix, 2d array of arbitrary size M * M
    b : Vector, 1d array of arbitrary size M * 1
    m : integer, m < M
    Outputs:
    Q : Matrix, 2d array of size M*(m+1)
    '''
    M = A.shape[0]
    Q = np.zeros((M, m + 1))
    q = b / np.linalg.norm(b)  # Normalize the initial vector b
    Q[:, 0] = q

    beta = 0
    q_prev = np.zeros(M)
    
    for j in range(m):
        v = A @ q  # Matrix-vector multiplication
        alpha = np.dot(q, v)  # Compute alpha
        v = v - alpha * q - beta * q_prev  # Orthogonalize
        beta = np.linalg.norm(v)  # Compute beta

        if beta < 1e-10:
            break  # If beta is zero, the process terminates early

        q_prev = q
        q = v / beta  # Normalize the vector
        Q[:, j + 1] = q

    return Q


try:
    targets = process_hdf5_to_tuple('5.1', 3)
    target = targets[0]
    n = 7
    h = 1.0/n
    diagonal = [2/h for i in range(n)]
    diagonal_up = [-1/h for i in range(n-1)]
    diagonal_down = [-1/h for i in range(n-1)]
    A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
    b = np.array([0.1,0.1,0.0,0.1,0.0,0.1,0.1])
    m = 5
    assert np.allclose(lanczos(A,b,m), target)

    target = targets[1]
    n = 7
    h = 1.0/n
    diagonal = [1/h for i in range(n)]
    diagonal_up = [-0.9/h for i in range(n-1)]
    diagonal_down = [-0.9/h for i in range(n-1)]
    A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
    b = np.array([0.1,10.1,0.0,0.5,0.2,0.3,0.5])
    m = 5
    assert np.allclose(lanczos(A,b,m), target)

    target = targets[2]
    n = 7
    h = 1.0/n
    diagonal = [1/h for i in range(n)]
    diagonal_up = [-9/h for i in range(n-1)]
    diagonal_down = [-9/h for i in range(n-1)]
    A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
    A[:, 0] = 0
    A[0, :] = 0
    A[0, 0] = 1/h
    b = np.array([0.1,0.1,0.0,10,0.0,0.1,0.1])
    m = 4
    assert np.allclose(lanczos(A,b,m), target)

