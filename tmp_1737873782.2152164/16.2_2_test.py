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
    k = num_eigenvalues * 2  # number of initial guess vectors, typically 2-3 times the number of eigenvalues
    
    # Step 1: Initialize guess vectors and other necessary matrices
    V = np.zeros((n, k))  # to hold the orthonormal basis of the Krylov subspace
    t = np.eye(n, k)  # initial unit vectors as guess
    I = np.eye(n)  # identity matrix for dimension of A

    # Step 2: Start with the first guess vector(s)
    V[:, :k] = t[:, :k]
    
    current_eigenvalues = np.zeros(num_eigenvalues)
    previous_eigenvalues = np.ones(num_eigenvalues) * np.inf

    # Step 3: Iteratively refine the subspace until convergence
    while np.max(np.abs(current_eigenvalues - previous_eigenvalues)) > threshold:
        # Step 3.1: Project matrixA onto the subspace
        AV = np.dot(matrixA, V)
        T = np.dot(V.T, AV)  # small projected matrix
        
        # Step 3.2: Solve the eigenvalue problem for the projected matrix
        eigvals, eigvecs = np.linalg.eigh(T)
        
        # Step 3.3: Update eigenvalues
        previous_eigenvalues = current_eigenvalues
        current_eigenvalues = eigvals[:num_eigenvalues]
        
        # Step 3.4: Expand the subspace with new directions
        for i in range(num_eigenvalues):
            residual = AV @ eigvecs[:, i] - eigvals[i] * (V @ eigvecs[:, i])
            if np.linalg.norm(residual) > threshold:
                # Preconditioner can be applied here if necessary
                correction_vector = np.linalg.solve(matrixA - eigvals[i] * I, residual)
                correction_vector /= np.linalg.norm(correction_vector)
                
                # Orthonormalize against the existing basis
                for j in range(V.shape[1]):
                    correction_vector -= np.dot(correction_vector, V[:, j]) * V[:, j]
                
                correction_vector /= np.linalg.norm(correction_vector)
                
                # Add the correction vector to the basis
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