import numpy as np



# Background: The QR factorization of a matrix A is a decomposition of A into a product A = QR,
# where Q is an orthogonal matrix and R is an upper triangular matrix. The Householder transformation
# is a method used to perform QR factorization. It involves a series of reflections to zero out
# the sub-diagonal elements of A, transforming it into an upper triangular matrix R. The matrix R
# is obtained by applying these transformations to A. The Householder transformation is particularly
# useful for numerical stability and efficiency in QR factorization.


def householder(A):
    '''Inputs:
    A : Matrix of size m*n, m>=n
    Outputs:
    A : Matrix of size m*n
    '''
    m, n = A.shape
    R = A.copy()
    
    for k in range(n):
        # Create the vector x
        x = R[k:, k]
        
        # Compute the norm of x
        norm_x = np.linalg.norm(x)
        
        # Create the vector e1 (first basis vector)
        e1 = np.zeros_like(x)
        e1[0] = 1
        
        # Compute the Householder vector v
        v = x + np.sign(x[0]) * norm_x * e1
        v = v / np.linalg.norm(v)
        
        # Apply the Householder transformation to R
        R[k:, k:] -= 2 * np.outer(v, np.dot(v, R[k:, k:]))
    
    # Return the upper triangular matrix R
    return np.triu(R)


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
