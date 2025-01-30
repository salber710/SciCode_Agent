from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import math
import numpy as np


def init_matrix(dim, noise):
    '''Generate a symmetric matrix with increasing values along its diagonal.
    Inputs:
    - dim: The dimension of the matrix (int).
    - noise: Noise level (float).
    Output:
    - A: a 2D array where each element is a float, representing the symmetric matrix.
    '''
    
    # Step 1: Create a diagonal matrix with increasing values along the diagonal
    A = np.zeros((dim, dim))
    for i in range(dim):
        A[i, i] = i + 1
    
    # Step 2: Add noise to the matrix
    noise_matrix = noise * np.random.randn(dim, dim)
    A += noise_matrix
    
    # Step 3: Symmetrize the matrix
    A = (A + A.T) / 2
    
    return A



def davidson_solver(matrixA, num_eigenvalues, threshold):
    '''Implements the Davidson algorithm to compute the first few eigenvalues of a symmetric matrix.
    Inputs:
    - matrixA: Symmetric matrix (2D array of float).
    - num_eigenvalues: Number of lowest eigenvalues to compute (int).
    - threshold: Convergence threshold for the algorithm (float).
    Output:
    - current_eigenvalues: computed eigenvalues (1D array of float).
    '''
    
    n = matrixA.shape[0]
    
    # Initial guess vectors and variables
    k = num_eigenvalues * 2  # A rule of thumb is to start with twice the number of desired eigenvalues
    V = np.zeros((n, k))  # Storage for guess vectors
    t = np.eye(n, k)  # Initial guess vectors (unit vectors)
    I = np.eye(n)  # Identity matrix
    
    # Initialize the subspace
    V[:, :k] = t
    
    # Placeholder for eigenvalues
    current_eigenvalues = np.zeros(num_eigenvalues)
    
    for iteration in range(100):  # Limit iterations to prevent infinite loops
        # Project A onto the subspace V
        T = np.dot(V.T, np.dot(matrixA, V))
        
        # Solve the small eigenproblem
        eigenvalues, eigenvectors = np.linalg.eigh(T)
        
        # Select the smallest eigenvalues
        current_eigenvalues = eigenvalues[:num_eigenvalues]
        current_eigenvectors = eigenvectors[:, :num_eigenvalues]
        
        # Compute the residuals
        residuals = []
        for i in range(num_eigenvalues):
            vi = np.dot(V, current_eigenvectors[:, i])  # The Ritz vector
            Av = np.dot(matrixA, vi)
            residual = Av - current_eigenvalues[i] * vi
            residuals.append(residual)
        
        # Check convergence
        max_residual_norm = max(np.linalg.norm(residual) for residual in residuals)
        if max_residual_norm < threshold:
            break
        
        # Update the subspace V with new vectors
        for i in range(num_eigenvalues):
            residual = residuals[i]
            correction_vector = residual / (current_eigenvalues[i] - np.diag(matrixA))
            correction_vector /= np.linalg.norm(correction_vector)
            
            # Orthogonalize against existing V
            for j in range(V.shape[1]):
                correction_vector -= np.dot(V[:, j].T, correction_vector) * V[:, j]
            
            # Normalize
            correction_vector /= np.linalg.norm(correction_vector)
            
            # Add to subspace
            V[:, k % V.shape[1]] = correction_vector
            k += 1

    return current_eigenvalues


try:
    targets = process_hdf5_to_tuple('16.2', 3)
    target = targets[0]
    np.random.seed(0)
    assert np.allclose(davidson_solver(init_matrix(100, 0.0),2,1e-8), target)

    target = targets[1]
    np.random.seed(1)
    assert np.allclose(davidson_solver(init_matrix(100, 0.0001), 5, 1e-8), target)

    target = targets[2]
    np.random.seed(2)
    assert np.allclose(davidson_solver(init_matrix(1000, 0.00001), 8, 1e-8), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e