import numpy as np



# Background: 
# QR factorization is a method used in linear algebra to decompose a matrix A into a product of an orthogonal matrix Q and an upper triangular matrix R. 
# For a given matrix A of size m*n with m >= n, the process involves using Householder transformations or Givens rotations to iteratively zero out the elements below the diagonal, resulting in the upper triangular matrix R.
# Householder transformations are a common method for QR factorization and they work by reflecting a vector about a plane or hyperplane. 
# In the context of QR factorization, a Householder transformation is used to zero out elements below the diagonal in a column of A.
# The matrix R is obtained as the final transformed version of A after applying these transformations.

def householder(A):
    '''Inputs:
    A : Matrix of size m*n, m>=n
    Outputs:
    A : Matrix of size m*n
    '''


    m, n = A.shape
    R = A.copy()

    for i in range(n):
        # Create the Householder vector
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1)
        u = x + e
        v = u / np.linalg.norm(u)

        # Apply the Householder transformation
        R[i:, i:] -= 2 * np.outer(v, v @ R[i:, i:])

    return R

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('74.1', 3)
target = targets[0]

A = np.array([[4, 1, 3], [2, 6, 8], [1, 4, 7]], dtype=float)
A_transformed = householder(A)
assert np.allclose(A_transformed, target)
target = targets[1]

A = np.array([[4, 1], [2, 6], [1, 4]], dtype=float)
A_transformed = householder(A)
assert np.allclose(A_transformed, target)
target = targets[2]

A = np.array([[10, 1], [7, 6], [1, 4], [5,5]], dtype=float)
A_transformed = householder(A)
assert np.allclose(A_transformed, target)
