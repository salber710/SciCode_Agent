import numpy as np



# Background: The Lanczos Iteration is an algorithm used to approximate eigenvalues and eigenvectors of a large symmetric matrix. 
# It is particularly useful for sparse matrices. The process generates an orthonormal basis for the Krylov subspace, which is 
# spanned by {b, Ab, A^2b, ..., A^(m-1)b}, where b is the initial vector. The algorithm iteratively constructs a tridiagonal 
# matrix T, and the columns of the matrix Q are the orthonormal basis vectors. The orthonormalization is typically done using 
# the Gram-Schmidt process. The matrix Q will have orthonormal columns, and the product Q^T * A * Q will approximate the 
# tridiagonal matrix T.


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
    beta = np.zeros(m+1)
    
    # Normalize the initial vector b
    Q[:, 0] = b / np.linalg.norm(b)
    
    for j in range(m):
        # Compute A * q_j
        v = A @ Q[:, j]
        
        # Orthogonalize against the previous vector
        alpha[j] = np.dot(Q[:, j], v)
        v = v - alpha[j] * Q[:, j] - beta[j] * Q[:, j-1] if j > 0 else v - alpha[j] * Q[:, j]
        
        # Compute the next beta
        beta[j+1] = np.linalg.norm(v)
        
        # Normalize to get the next q vector
        if beta[j+1] != 0:
            Q[:, j+1] = v / beta[j+1]
        else:
            # If beta is zero, the Krylov subspace has been exhausted
            break
    
    return Q


from scicode.parse.parse import process_hdf5_to_tuple

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
