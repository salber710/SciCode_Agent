from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import numpy.linalg as la
from scipy import signal


def center(X, divide_sd=True):
    '''Center the input matrix X and optionally scale it by the standard deviation.
    Args:
        X (np.ndarray): The input matrix of shape (nmix, time).
        divide_sd (bool): If True, divide by the standard deviation. Defaults to True.
    Returns:
        np.ndarray: The centered (and optionally scaled) matrix of the same shape as the input.
    '''
    # Calculate the mean along each row (axis=1) and keep the dimensions for broadcasting
    mean_X = np.mean(X, axis=1, keepdims=True)
    # Center the matrix by subtracting the mean
    centered_X = X - mean_X

    if divide_sd:
        # Calculate the standard deviation along each row (axis=1) and keep the dimensions for broadcasting
        std_X = np.std(X, axis=1, keepdims=True)
        # Avoid division by zero in case of zero standard deviation
        std_X[std_X == 0] = 1
        # Scale the centered matrix by dividing by the standard deviation
        centered_X = centered_X / std_X
    
    return centered_X



def whiten(X):
    '''Whiten matrix X
    Args: 
        X (np.array): mixture matrix. Shape (nmix, time)
    Return:
        Z (np.array): whitened matrix. Shape (nmix, time)
    '''
    # Step 1: Center the data
    X_centered = center(X, divide_sd=False)
    
    # Step 2: Compute the covariance matrix of the centered data
    cov_X = np.cov(X_centered, rowvar=True)
    
    # Step 3: Perform eigenvalue decomposition of the covariance matrix
    eigvals, eigvecs = la.eigh(cov_X)
    
    # Step 4: Compute the whitening matrix
    # We want to scale the eigenvectors by the inverse square root of the eigenvalues
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    whitening_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T
    
    # Step 5: Whiten the data using the whitening matrix
    Z = whitening_matrix @ X_centered
    
    return Z


try:
    targets = process_hdf5_to_tuple('31.2', 3)
    target = targets[0]
    def test_identity(A):
        return np.allclose(A, np.eye(A.shape[0]))
    Z = whiten(np.array([[ -4.        ,  -1.25837414,  -4.2834508 ,   4.22567322,
              1.43150983,  -6.28790332],
           [ -4.        ,  -3.22918707,  -6.3417254 ,   6.31283661,
              3.31575491,  -8.14395166],
           [ -8.        ,  -0.48756122,  -6.62517619,   6.53850983,
              0.74726474, -10.43185497]]))
    s, v = target
    assert test_identity(np.cov(Z)) == s and np.allclose(Z, v)

    target = targets[1]
    def test_identity(A):
        return np.allclose(A, np.eye(A.shape[0]))
    Z = whiten(np.array([[ -4.        ,  -4.10199583,  -0.70436724,  -2.02846889,
              2.84962972,  -1.19342653,   5.76905316,  -6.28790332],
           [ -4.        ,  -6.47956934,   1.79067353,  -4.29994873,
              4.71052915,  -2.73957041,   7.31309801,  -8.14395166],
           [ -8.        ,  -6.58156517,  -2.91369371,  -2.32841762,
              3.56015887,   0.06700306,   9.08215117, -10.43185497]]))
    s, v = target
    assert test_identity(np.cov(Z)) == s and np.allclose(Z, v)

    target = targets[2]
    def test_identity(A):
        return np.allclose(A, np.eye(A.shape[0]))
    Z = whiten(np.array([[-4.00000000e+00,  6.08976682e+00, -1.80018426e-01,
             2.52000394e+00, -8.19025595e-01,  2.06616123e+00,
            -4.27972909e+00, -4.34384652e+00, -1.14726131e-01,
            -6.28790332e+00],
           [-4.00000000e+00,  7.60043896e+00, -1.97889810e+00,
             4.92666864e+00, -3.18729058e+00,  3.81085839e+00,
            -5.80653121e+00, -6.28303437e+00,  1.38708138e+00,
            -8.14395166e+00],
           [-8.00000000e+00,  9.69020578e+00,  1.84108347e+00,
             3.44667258e+00, -6.31617101e-03,  1.87701963e+00,
            -6.08626030e+00, -6.62688090e+00, -2.72764475e+00,
            -1.04318550e+01]]))
    s, v = target
    assert test_identity(np.cov(Z)) == s and np.allclose(Z, v)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e