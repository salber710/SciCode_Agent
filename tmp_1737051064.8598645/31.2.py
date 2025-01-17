import numpy as np
import numpy.linalg as la
from scipy import signal

# Background: Standardizing a matrix involves centering the data by subtracting the mean of each row and optionally scaling it by dividing by the standard deviation of each row. This process is crucial in data preprocessing to ensure that each feature contributes equally to the analysis, especially in algorithms sensitive to the scale of the data. Centering the data helps in removing biases due to different scales, and scaling by the standard deviation ensures that the data has unit variance, which is often a requirement for many machine learning algorithms.


def center(X, divide_sd=True):
    '''Center the input matrix X and optionally scale it by the standard deviation.
    Args:
        X (np.ndarray): The input matrix of shape (nmix, time).
        divide_sd (bool): If True, divide by the standard deviation. Defaults to True.
    Returns:
        np.ndarray: The centered (and optionally scaled) matrix of the same shape as the input.
    '''
    # Calculate the mean of each row
    row_means = np.mean(X, axis=1, keepdims=True)
    
    # Center the matrix by subtracting the row means
    D = X - row_means
    
    if divide_sd:
        # Calculate the standard deviation of each row
        row_stds = np.std(X, axis=1, keepdims=True)
        
        # Avoid division by zero by setting zero stds to 1
        row_stds[row_stds == 0] = 1
        
        # Scale the centered matrix by dividing by the row standard deviations
        D = D / row_stds
    
    return D



# Background: Whitening a matrix is a preprocessing step that transforms the data such that its covariance matrix becomes the identity matrix. This process is crucial in many signal processing and machine learning tasks, as it removes correlations between features and ensures that each feature has unit variance. Whitening typically involves centering the data, computing the covariance matrix, and then using the eigenvalue decomposition of the covariance matrix to transform the data. The transformation ensures that the resulting data is uncorrelated and standardized, which can improve the performance of algorithms that assume input features are independent and identically distributed.



def whiten(X):
    '''Whiten matrix X
    Args: 
        X (np.array): mixture matrix. Shape (nmix, time)
    Return:
        Z (np.array): whitened matrix. Shape (nmix, time)
    '''
    # Center the matrix X along the rows
    row_means = np.mean(X, axis=1, keepdims=True)
    X_centered = X - row_means
    
    # Compute the covariance matrix of the centered data
    covariance_matrix = np.cov(X_centered, rowvar=True, bias=True)
    
    # Perform eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = la.eigh(covariance_matrix)
    
    # Construct the whitening matrix
    whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # Whiten the centered data
    Z = whitening_matrix @ X_centered
    
    return Z


from scicode.parse.parse import process_hdf5_to_tuple

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
