from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The Lanczos Iteration is an algorithm used to approximate the eigenvalues and eigenvectors of a large symmetric matrix. 
# It is particularly useful for sparse matrices. The process involves constructing an orthonormal basis for the Krylov subspace 
# generated by the matrix A and a starting vector b. The algorithm iteratively builds this basis using a process similar to the 
# Gram-Schmidt orthogonalization. The result is a tridiagonal matrix T, which is much smaller and easier to handle than the original 
# matrix A. The columns of the matrix Q form an orthonormal basis for the Krylov subspace, and the matrix T is the projection of A 
# onto this subspace.


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
    alpha = np.zeros(m)
    beta = np.zeros(m)
    
    # Normalize the initial vector b
    Q[:, 0] = b / np.linalg.norm(b)
    
    for j in range(m):
        # Compute the matrix-vector product
        v = A @ Q[:, j]
        
        # Compute the alpha coefficient
        alpha[j] = np.dot(Q[:, j], v)
        
        # Orthogonalize against the previous vector
        v = v - alpha[j] * Q[:, j]
        
        if j > 0:
            v = v - beta[j-1] * Q[:, j-1]
        
        # Compute the beta coefficient
        beta[j] = np.linalg.norm(v)
        
        # Normalize the new vector and add it to Q
        if beta[j] != 0:
            Q[:, j+1] = v / beta[j]
        else:
            # If beta is zero, the Krylov subspace has been fully spanned
            break
    
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e