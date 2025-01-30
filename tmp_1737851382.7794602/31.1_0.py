import numpy as np
import numpy.linalg as la
from scipy import signal



# Background: Standardizing a matrix involves centering the data by subtracting the mean of each row and optionally scaling it by dividing by the standard deviation of each row. This process is crucial in data preprocessing to ensure that each feature contributes equally to the analysis, especially in algorithms sensitive to the scale of the data. Centering the data helps in removing bias, and scaling it by the standard deviation ensures that the data has unit variance, which is often a requirement for many machine learning algorithms.


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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('31.1', 3)
target = targets[0]

assert np.allclose(center(np.array([[ -4.        ,  -1.25837414,  -4.2834508 ,   4.22567322,
          1.43150983,  -6.28790332],
       [ -4.        ,  -3.22918707,  -6.3417254 ,   6.31283661,
          3.31575491,  -8.14395166],
       [ -8.        ,  -0.48756122,  -6.62517619,   6.53850983,
          0.74726474, -10.43185497]])), target)
target = targets[1]

assert np.allclose(center(np.array([[ -4.        ,  -4.10199583,  -0.70436724,  -2.02846889,
          2.84962972,  -1.19342653,   5.76905316,  -6.28790332],
       [ -4.        ,  -6.47956934,   1.79067353,  -4.29994873,
          4.71052915,  -2.73957041,   7.31309801,  -8.14395166],
       [ -8.        ,  -6.58156517,  -2.91369371,  -2.32841762,
          3.56015887,   0.06700306,   9.08215117, -10.43185497]])), target)
target = targets[2]

assert np.allclose(center(np.array([[-4.00000000e+00,  6.08976682e+00, -1.80018426e-01,
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
        -1.04318550e+01]])), target)
