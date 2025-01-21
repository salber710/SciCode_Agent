import numpy as np



# Background: The Lanczos Iteration is an algorithm to approximate the eigenvalues and eigenvectors 
# of a large symmetric matrix. It is especially useful in numerical linear algebra for solving 
# problems involving large sparse matrices. The Lanczos process generates a sequence of 
# orthogonal vectors and a tridiagonal matrix that approximates the original matrix. 
# The process starts with an initial vector and iteratively applies the matrix to build 
# an orthonormal basis for the Krylov subspace. The orthonormal vectors are stored in the 
# columns of matrix Q. The size of Q is determined by the number of iterations m.


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
        z = A @ q
        if j > 0:
            z = z - beta * Q[:, j-1]
        alpha = np.dot(q, z)
        z = z - alpha * q
        beta = np.linalg.norm(z)
        if beta > 0:
            q = z / beta
        Q[:, j+1] = q
    
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
