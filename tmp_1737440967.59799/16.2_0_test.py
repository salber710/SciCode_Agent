import math
import numpy as np

# Background: A symmetric matrix is a square matrix that is equal to its transpose, meaning the element at row i, column j
# is equal to the element at row j, column i. When constructing such a matrix with increasing values along its diagonal, 
# we ensure that the primary diagonal elements increment from a base value (e.g., 1, 2, 3, ..., n for an n x n matrix).
# To incorporate randomness, we multiply each element by a normally distributed random number. In numpy, the normal distribution 
# can be generated using `np.random.normal()`. The symmetrization process involves averaging the matrix with its transpose.
# This ensures that any asymmetries introduced by the random noise are corrected, preserving the symmetric property.

def init_matrix(dim, noise):
    '''Generate a symmetric matrix with increasing values along its diagonal.
    Inputs:
    - dim: The dimension of the matrix (int).
    - noise: Noise level (float).
    Output:
    - A: a 2D array where each element is a float, representing the symmetric matrix.
    '''

    # Initialize the matrix with zeros
    M = np.zeros((dim, dim))
    
    # Fill the diagonal with increasing values
    for i in range(dim):
        M[i, i] = i + 1
    
    # Introduce noise by multiplying with a normally distributed random number
    # The noise is scaled by the input noise parameter
    noise_matrix = np.random.normal(loc=1.0, scale=noise, size=(dim, dim))
    M = M * noise_matrix
    
    # Symmetrize the matrix by averaging it with its transpose
    A = (M + M.T) / 2
    
    return A



# Background: Davidson's method is an iterative algorithm used to find a few of the lowest eigenvalues and their corresponding eigenvectors of a large, symmetric matrix. 
# It is particularly useful for quantum chemistry and other fields that require solutions of large eigenvalue problems. The method iteratively refines a subspace 
# to approximate the eigenvalues and eigenvectors. Starting with an initial guess, it expands the subspace by adding correction vectors calculated based on the residuals
# of the current approximation. The process continues until the changes in the eigenvalues are smaller than a specified convergence threshold, indicating that the solution
# has stabilized. The convergence threshold is a measure of how close the estimated eigenvalues should be to the actual eigenvalues for the algorithm to stop.
# The Davidson algorithm is efficient for matrices that are too large to handle with direct methods like QR decomposition.


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
    # Initial guess for the subspace (using identity matrix subspace)
    subspace_dim = num_eigenvalues + 1
    V = np.eye(n, subspace_dim)
    
    current_eigenvalues = np.zeros(num_eigenvalues)
    previous_eigenvalues = np.ones(num_eigenvalues) * np.inf
    
    while np.max(np.abs(current_eigenvalues - previous_eigenvalues)) > threshold:
        # Compute subspace matrix B = V^T * A * V
        B = np.dot(V.T, np.dot(matrixA, V))
        
        # Diagonalize B
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # Sort eigenvalues and eigenvectors
        idx = eigvals.argsort()[:num_eigenvalues]
        current_eigenvalues = eigvals[idx]
        current_eigenvectors = eigvecs[:, idx]
        
        # Update the eigenvectors in the original space
        V_updated = np.dot(V, current_eigenvectors)
        
        # Calculate residuals
        residuals = np.dot(matrixA, V_updated) - np.dot(V_updated, np.diag(current_eigenvalues))
        
        # Normalize residuals and find correction vectors
        for i in range(num_eigenvalues):
            resid_norm = np.linalg.norm(residuals[:, i])
            if resid_norm > threshold:
                residuals[:, i] /= resid_norm
                # Add the new direction to the subspace
                V = np.column_stack((V, residuals[:, i]))
        
        previous_eigenvalues = current_eigenvalues
    
    return current_eigenvalues

from scicode.parse.parse import process_hdf5_to_tuple
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
