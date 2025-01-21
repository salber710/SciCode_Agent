
# Background: Davidson's method is an iterative algorithm used to find a few of the lowest eigenvalues 
# and corresponding eigenvectors of large symmetric matrices. It is particularly useful when dealing 
# with sparse matrices or when only a small number of eigenvalues are needed. The method builds a 
# subspace iteratively, solving the eigenvalue problem in this reduced space, and refines it by adding 
# correction vectors. The convergence of the algorithm is typically controlled by a threshold parameter 
# which determines the accuracy of the eigenvalue approximation.

import math
import numpy as np

def davidson_solver(matrixA, num_eigenvalues, threshold):
    '''Implements the Davidson algorithm to compute the first few eigenvalues of a symmetric matrix.
    Inputs:
    - matrixA: Symmetric matrix (2D array of float).
    - num_eigenvalues: Number of lowest eigenvalues to compute (int).
    - threshold: Convergence threshold for the algorithm (float).
    Output:
    - current_eigenvalues: computed eigenvalues (1D array of float).
    '''
    
    dim = matrixA.shape[0]
    subspace_size = min(2 * num_eigenvalues, dim)
    V = np.random.rand(dim, subspace_size)  # Initial guess for subspace
    V, _ = np.linalg.qr(V)  # Orthonormalize the initial subspace
    
    conv = False
    current_eigenvalues = np.zeros(num_eigenvalues)
    
    while not conv:
        # Project A into the subspace
        AV = np.dot(matrixA, V)
        subspace_matrix = np.dot(V.T, AV)
        
        # Solve the reduced eigenvalue problem
        eigvals, eigvecs = np.linalg.eigh(subspace_matrix)
        
        # Calculate residuals
        conv = True
        for i in range(num_eigenvalues):
            r = AV @ eigvecs[:, i] - eigvals[i] * (V @ eigvecs[:, i])
            residual_norm = np.linalg.norm(r)
            if residual_norm > threshold:
                conv = False
                # Update the subspace with a correction vector
                correction_vector = r / (eigvals[i] - np.diag(matrixA))
                correction_vector -= np.dot(V, np.dot(V.T, correction_vector))
                V = np.column_stack((V, correction_vector))
                if V.shape[1] > subspace_size:
                    V = V[:, :subspace_size]
                V, _ = np.linalg.qr(V)
                break
        
        # Update the eigenvalues
        if conv:
            current_eigenvalues = eigvals[:num_eigenvalues]
    
    return current_eigenvalues
