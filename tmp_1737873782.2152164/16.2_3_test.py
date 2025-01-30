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
    # Dimensions of the matrix
    n = matrixA.shape[0]
    
    # Initial guess for the subspace dimension
    k = min(2 * num_eigenvalues, n)
    
    # Start with a set of orthonormal vectors (identity matrix columns)
    V = np.eye(n, k)
    
    # Allocate space for the guess vectors
    T = np.zeros((n, num_eigenvalues))
    
    # Initial eigenvalues and eigenvectors
    current_eigenvalues = np.zeros(num_eigenvalues)
    
    while True:
        # Subspace projection
        AV = matrixA @ V
        H = V.T @ AV  # Projected matrix
        
        # Solve the eigenproblem in the subspace
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # Update the Ritz vectors
        T = V @ eigvecs[:, :num_eigenvalues]
        
        # Check convergence
        is_converged = True
        for i in range(num_eigenvalues):
            # Compute the residual vector
            residual = AV @ T[:, i] - eigvals[i] * T[:, i]
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm > threshold:
                is_converged = False
                # Update the subspace with the preconditioned residual
                correction = residual / (eigvals[i] - np.diag(matrixA))
                V = np.column_stack((V, correction / np.linalg.norm(correction)))
        
        if is_converged:
            break
        
        # Orthonormalize V
        V, _ = np.linalg.qr(V)
    
    # Return the converged eigenvalues
    current_eigenvalues = eigvals[:num_eigenvalues]
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