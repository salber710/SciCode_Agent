
# Background: Davidson's method is an iterative algorithm used to find a few of the lowest eigenvalues and corresponding eigenvectors of a large sparse symmetric matrix. 
# It is particularly useful in quantum chemistry and physics for solving large eigenvalue problems. The method starts with an initial guess of the eigenvectors and 
# iteratively refines them. The algorithm builds a subspace and projects the matrix onto this subspace to find approximate eigenvalues and eigenvectors. 
# The convergence is controlled by a threshold, and the process continues until the change in eigenvalues is below this threshold.

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
    
    # Check if the matrix is symmetric
    if not np.allclose(matrixA, matrixA.T):
        raise ValueError("Matrix is not symmetric.")
    
    # Initial guess: start with a random vector
    dim = matrixA.shape[0]
    V = np.random.rand(dim, num_eigenvalues)
    
    # Orthogonalize the initial guess
    V, _ = np.linalg.qr(V)
    
    # Initialize the subspace matrix
    T = np.zeros((num_eigenvalues, num_eigenvalues))
    
    # Initialize eigenvalues
    current_eigenvalues = np.zeros(num_eigenvalues)
    
    # Iterative process
    while True:
        # Project matrixA onto the subspace spanned by V
        AV = np.dot(matrixA, V)
        T = np.dot(V.T, AV)
        
        # Solve the small eigenvalue problem
        eigvals, eigvecs = np.linalg.eigh(T)
        
        # Update the eigenvalues
        new_eigenvalues = eigvals[:num_eigenvalues]
        
        # Check for convergence
        if np.all(np.abs(new_eigenvalues - current_eigenvalues) < threshold):
            break
        
        # Update current eigenvalues
        current_eigenvalues = new_eigenvalues
        
        # Compute the residuals
        residuals = AV - np.dot(V, np.dot(T, eigvecs))
        
        # Normalize the residuals
        for i in range(num_eigenvalues):
            residuals[:, i] /= np.linalg.norm(residuals[:, i])
        
        # Expand the subspace
        V = np.hstack((V, residuals))
        
        # Re-orthogonalize V
        V, _ = np.linalg.qr(V)
        
        # Reduce V to the original subspace size
        V = V[:, :num_eigenvalues]
    
    return current_eigenvalues
