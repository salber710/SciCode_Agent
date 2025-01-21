import numpy as np



# Background: The QR factorization of a matrix A is a decomposition of A into a product A = QR, where Q is an orthogonal matrix and R is an upper triangular matrix. The Householder transformation is a common method for computing the QR factorization. In this method, A is transformed into R by a series of orthogonal transformations. Specifically, for each column of A, a Householder vector is used to zero out the below-diagonal elements, effectively transforming A into R. The matrix R is the upper triangular part of the transformed matrix A after applying these transformations.


def householder(A):
    '''Inputs:
    A : Matrix of size m*n, m>=n
    Outputs:
    A : Matrix of size m*n
    '''
    m, n = A.shape
    R = A.copy().astype(float)

    for k in range(n):
        # Create the Householder vector for the k-th column
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        
        # Apply the transformation to zero out below-diagonal elements
        R[k:, k:] -= 2 * np.outer(v, v.dot(R[k:, k:]))

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
