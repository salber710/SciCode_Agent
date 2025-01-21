from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The Lanczos Iteration is an algorithm used to find a tridiagonal matrix representation of a symmetric matrix A.
# This method is particularly useful in numerical linear algebra for approximating eigenvalues and eigenvectors of large sparse
# symmetric matrices. The process involves constructing an orthonormal set of basis vectors for the Krylov subspace generated
# by A and an initial vector b. These basis vectors are stored in the matrix Q. The algorithm iteratively refines the vectors,
# ensuring orthogonality through reorthogonalization. This is crucial to maintain numerical stability and accuracy. The iteration
# stops after 'm' steps, where 'm' is less than the size of the matrix A. The resulting matrix Q will have orthonormal columns,
# forming a basis for the Krylov subspace.


def lanczos(A, b, m):
    '''Inputs:
    A : Matrix, 2d array of arbitrary size M * M
    b : Vector, 1d array of arbitrary size M * 1
    m : integer, m < M
    Outputs:
    Q : Matrix, 2d array of size M*(m+1)
    '''
    M = A.shape[0]
    Q = np.zeros((M, m+1))
    q = b / np.linalg.norm(b)
    Q[:, 0] = q
    beta = 0
    
    for j in range(m):
        if j == 0:
            r = np.dot(A, q)
        else:
            r = np.dot(A, q) - beta * Q[:, j-1]
        
        alpha = np.dot(q, r)
        r = r - alpha * q
        
        # Reorthogonalize to maintain orthogonality
        for k in range(j):
            r -= np.dot(Q[:, k], r) * Q[:, k]
        
        beta = np.linalg.norm(r)
        
        if beta < 1e-10:  # Check for convergence
            break
        
        q = r / beta
        Q[:, j+1] = q
    
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

