from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import numpy.linalg as la
from scipy import signal


def center(X, divide_sd=True):
    # Calculate the mean and standard deviation using a loop for a different approach
    nmix, time = X.shape
    D = np.empty_like(X)
    for i in range(nmix):
        mean = np.mean(X[i])
        std = np.std(X[i])
        std = std if std != 0 else 1
        D[i] = (X[i] - mean) / std if divide_sd else (X[i] - mean)
    return D



def whiten(X):
    '''Whiten matrix X
    Args: 
        X (np.array): mixture matrix. Shape (nmix, time)
    Return:
        Z (np.array): whitened matrix. Shape (nmix, time)
    '''
    # Center the matrix X along the rows
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    
    # Compute the covariance matrix of the centered data
    cov_matrix = np.cov(X_centered, rowvar=True)
    
    # Use the Jordan decomposition to find the whitening matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Compute the inverse square root of the diagonal matrix of eigenvalues
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    
    # Construct the whitening matrix using the Jordan decomposition components
    whitening_matrix = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    
    # Apply the whitening matrix to the centered data
    Z = whitening_matrix @ X_centered
    
    return Z



# Background: Independent Component Analysis (ICA) is a computational technique for separating a multivariate signal into additive, independent components. It is a special case of blind source separation. The goal of ICA is to find a linear transformation that makes the components as statistically independent from each other as possible. The FastICA algorithm is a popular method for performing ICA, which involves whitening the data and then iteratively estimating the unmixing matrix using a fixed-point iteration scheme. The function g(x) = tanh(x) is used as a non-linear function to approximate the negentropy, and its derivative dg(x) = 1 - g(x)^2 is used in the Newton's method update step. The algorithm iterates until convergence, which is determined by the change in the estimated unmixing vector.

def ica(X, cycles, tol):
    '''Perform independent component analysis 
    Args:
        X (np.array): mixture matrix. Shape (nmix, time)
        cycles (int): number of max possible iterations 
        tol (float): convergence tolerance
    Returns:
        S_hat (np.array): predicted independent sources. Shape (nmix, time)
    '''
    # Whiten the input matrix X
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    cov_matrix = np.cov(X_centered, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    whitening_matrix = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    X_whitened = whitening_matrix @ X_centered

    nmix, _ = X.shape
    W = np.zeros((nmix, nmix))

    for i in range(nmix):
        w = np.random.rand(nmix)
        w /= np.linalg.norm(w)

        for _ in range(cycles):
            w_new = np.mean(X_whitened * np.tanh(X_whitened.T @ w), axis=1) - np.mean(1 - np.tanh(X_whitened.T @ w)**2) * w
            w_new /= np.linalg.norm(w_new)

            if i > 0:
                w_new -= W[:i].T @ (W[:i] @ w_new)

            if np.abs(np.abs(w @ w_new) - 1) < tol:
                break

            w = w_new

        W[i, :] = w

    S_hat = W @ X_whitened
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