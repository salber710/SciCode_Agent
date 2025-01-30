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
    # Center the matrix X along each row
    centered_X = center(X, divide_sd=True)
    
    # Calculate the covariance matrix of the centered data
    cov_X = np.cov(centered_X, rowvar=True)
    
    # Perform eigenvalue decomposition of the covariance matrix
    eigvals, eigvecs = la.eigh(cov_X)
    
    # Form the whitening matrix using the eigenvectors and eigenvalues
    whitening_matrix = np.dot(eigvecs, np.diag(1.0 / np.sqrt(eigvals)))
    
    # Whiten the data
    Z = np.dot(whitening_matrix, centered_X)
    
    return Z






def ica(X, cycles, tol):
    '''Perform independent component analysis 
    Args:
        X (np.array): mixture matrix. Shape (nmix, time)
        cycles (int): number of max possible iterations 
        tol (float): convergence tolerance
    Returns:
        S_hat (np.array): predicted independent sources. Shape (nmix, time)
    '''
    def g(x):
        return np.tanh(x)

    def dg(x):
        return 1 - np.tanh(x)**2

    # Whitening the data
    X_whitened = whiten(X)
    
    nmixtures, time = X.shape
    W = np.zeros((nmixtures, nmixtures))
    
    for i in range(nmixtures):
        w = np.random.rand(nmixtures)
        w = w / np.linalg.norm(w)  # Normalize the initial vector

        for _ in range(cycles):
            w_new = np.mean(X_whitened * g(np.dot(w, X_whitened)), axis=1) - np.mean(dg(np.dot(w, X_whitened))) * w
            w_new = w_new / np.linalg.norm(w_new)  # Normalize the new vector

            if np.abs(np.abs(np.dot(w, w_new)) - 1) < tol:
                break
            w = w_new

        W[i, :] = w

    S_hat = np.dot(W, X_whitened)
    return S_hat


try:
    targets = process_hdf5_to_tuple('31.3', 4)
    target = targets[0]
    np.random.seed(0)
    assert np.allclose(ica(np.array([[ -4.        ,  -1.25837414,  -4.2834508 ,   4.22567322,
              1.43150983,  -6.28790332],
           [ -4.        ,  -3.22918707,  -6.3417254 ,   6.31283661,
              3.31575491,  -8.14395166],
           [ -8.        ,  -0.48756122,  -6.62517619,   6.53850983,
              0.74726474, -10.43185497]]), cycles=200, tol=1e-5), target)

    target = targets[1]
    np.random.seed(0)
    assert np.allclose(ica(np.array([[ -4.        ,  -4.10199583,  -0.70436724,  -2.02846889,
              2.84962972,  -1.19342653,   5.76905316,  -6.28790332],
           [ -4.        ,  -6.47956934,   1.79067353,  -4.29994873,
              4.71052915,  -2.73957041,   7.31309801,  -8.14395166],
           [ -8.        ,  -6.58156517,  -2.91369371,  -2.32841762,
              3.56015887,   0.06700306,   9.08215117, -10.43185497]]), cycles=200, tol=1e-5), target)

    target = targets[2]
    np.random.seed(0)
    assert np.allclose(ica(np.array([[-4.00000000e+00,  6.08976682e+00, -1.80018426e-01,
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
            -1.04318550e+01]]), cycles=200, tol=1e-5), target)

    target = targets[3]
    np.random.seed(0)
    def create_signals(N=2000):
        '''
        Load example data. 
        In this example, we use sinusoidal, square, and sawtooth signals as our independent sources.
        The matrix `A` that transform X to S is fixed to 
            A = np.array([
                [1, 1, 1],
                [0.5, 2, 1],
                [1.5, 1, 2]
                ])    
        Returns:
            X (np.array): mixture matrix. Shape (nmix, time)
            S (np.array): original independent sources. Shape (nmix, time)
        '''
        time = np.linspace(0, 8, N)
        s1 = np.sin(2*time) # sinusoidal
        s2 = 2*np.sign(np.sin(3*time)) # square signal
        s3 = 4*signal.sawtooth(2*np.pi*time)  # saw tooth signal
        S = np.array([s1, s2, s3])
        A = np.array([
            [1, 1, 1],
            [0.5, 2, 1],
            [1.5, 1, 2]
            ])
        X = A @ S
        return X, S
    X, S = create_signals(N=2000)
    S_hat = ica(X, cycles=200, tol=1e-5)
    assert np.allclose(S_hat, target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e