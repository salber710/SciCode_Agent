from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import math
import numpy as np


def init_matrix(dim, noise):
    # Create a matrix with increasing diagonal values using a logarithmic scale
    A = np.diag(np.log1p(np.arange(1, dim + 1)))
    
    # Generate a matrix of random numbers, each element being the product of a normally distributed random number and the logarithm of its column index
    random_matrix = np.random.normal(0, 1, (dim, dim)) * np.log1p(np.arange(1, dim + 1))
    
    # Scale the random matrix by the noise level
    A += noise * random_matrix
    
    # Symmetrize the matrix by averaging it with its transpose
    A = (A + A.T) / 2
    
    return A



# Background: Davidson's method is an iterative algorithm used to find a few of the lowest eigenvalues and corresponding eigenvectors of a large sparse symmetric matrix. 
# It is particularly useful in quantum chemistry and physics for solving large eigenvalue problems. The method starts with an initial guess for the eigenvectors and 
# iteratively refines them. The key idea is to expand a subspace in which the eigenproblem is solved, using corrections derived from the residuals of the current 
# approximate eigenvectors. The process continues until the change in eigenvalues between iterations is below a specified convergence threshold.


def davidson_solver(matrixA, num_eigenvalues, threshold):
    '''Implements the Davidson algorithm to compute the first few eigenvalues of a symmetric matrix.
    Inputs:
    - matrixA: Symmetric matrix (2D array of float).
    - num_eigenvalues: Number of lowest eigenvalues to compute (int).
    - threshold: Convergence threshold for the algorithm (float).
    Output:
    - current_eigenvalues: computed eigenvalues (1D array of float).
    '''
    
    # Initial setup
    dim = matrixA.shape[0]
    subspace_dim = min(2 * num_eigenvalues, dim)  # Initial subspace dimension
    V = np.random.rand(dim, subspace_dim)  # Random initial subspace
    V, _ = np.linalg.qr(V)  # Orthonormalize the subspace
    
    current_eigenvalues = np.zeros(num_eigenvalues)
    prev_eigenvalues = np.ones(num_eigenvalues) * np.inf
    
    while np.max(np.abs(current_eigenvalues - prev_eigenvalues)) > threshold:
        # Project matrixA into the subspace
        AV = matrixA @ V
        T = V.T @ AV  # Subspace matrix
        
        # Solve the eigenvalue problem in the subspace
        eigvals, eigvecs = np.linalg.eigh(T)
        
        # Sort eigenvalues and select the smallest ones
        idx = np.argsort(eigvals)[:num_eigenvalues]
        current_eigenvalues = eigvals[idx]
        current_eigenvectors = eigvecs[:, idx]
        
        # Compute the residuals
        residuals = AV @ current_eigenvectors - V @ (current_eigenvectors * current_eigenvalues)
        
        # Check convergence
        if np.max(np.abs(current_eigenvalues - prev_eigenvalues)) <= threshold:
            break
        
        # Update previous eigenvalues
        prev_eigenvalues = current_eigenvalues.copy()
        
        # Expand the subspace with the residuals
        for i in range(num_eigenvalues):
            r = residuals[:, i]
            if np.linalg.norm(r) > 1e-10:  # Avoid division by zero
                r /= (current_eigenvalues[i] - np.diag(matrixA))  # Preconditioner
                V = np.hstack((V, r.reshape(-1, 1)))
        
        # Re-orthonormalize the subspace
        V, _ = np.linalg.qr(V)
        
        # Limit the size of the subspace
        if V.shape[1] > subspace_dim:
            V = V[:, :subspace_dim]
    
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