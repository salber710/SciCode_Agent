import math
import numpy as np

# Background: A symmetric matrix is a square matrix that is equal to its transpose. 
# In this task, we need to create a symmetric matrix with increasing values along its diagonal.
# The diagonal elements will be integers from 1 to the dimension of the matrix.
# Each element of the matrix will be modified by a product of a normally distributed random number 
# and a user-provided noise level. The matrix is then symmetrized by averaging it with its transpose.
# This process ensures that the resulting matrix is symmetric and incorporates randomness influenced by the noise level.


def init_matrix(dim, noise):
    '''Generate a symmetric matrix with increasing values along its diagonal.
    Inputs:
    - dim: The dimension of the matrix (int).
    - noise: Noise level (float).
    Output:
    - A: a 2D array where each element is a float, representing the symmetric matrix.
    '''
    # Initialize a matrix with zeros
    matrix = np.zeros((dim, dim))
    
    # Set increasing values along the diagonal
    for i in range(dim):
        matrix[i, i] = i + 1
    
    # Modify each element by a product of a normally distributed random number and the noise level
    random_noise = np.random.normal(0, 1, (dim, dim))
    matrix += random_noise * noise
    
    # Symmetrize the matrix by averaging it with its transpose
    A = (matrix + matrix.T) / 2
    
    return A



# Background: Davidson's method is an iterative algorithm used to find a few of the lowest eigenvalues and corresponding eigenvectors of a large sparse symmetric matrix. 
# It is particularly useful in quantum chemistry and physics for solving large eigenvalue problems. The method starts with an initial guess for the eigenvectors and 
# iteratively refines them. The algorithm builds a subspace and projects the matrix onto this subspace to find approximate eigenvalues and eigenvectors. 
# The process is repeated, expanding the subspace until the eigenvalues converge to within a specified threshold. The convergence is typically measured by the 
# residual norm, which should be below the given threshold for the solution to be considered accurate.



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
    max_iterations = 1000
    subspace_dim = min(2 * num_eigenvalues, dim)
    
    # Initial guess for the subspace
    V = np.random.rand(dim, subspace_dim)
    V, _ = np.linalg.qr(V)  # Orthonormalize the initial subspace
    
    current_eigenvalues = np.zeros(num_eigenvalues)
    
    for iteration in range(max_iterations):
        # Project matrixA onto the subspace V
        AV = np.dot(matrixA, V)
        T = np.dot(V.T, AV)
        
        # Solve the eigenvalue problem in the subspace
        eigvals, eigvecs = np.linalg.eigh(T)
        
        # Select the desired number of eigenvalues
        current_eigenvalues = eigvals[:num_eigenvalues]
        current_eigvecs = eigvecs[:, :num_eigenvalues]
        
        # Compute the residuals
        residuals = AV @ current_eigvecs - V @ (current_eigvecs * current_eigenvalues)
        
        # Check convergence
        residual_norms = np.linalg.norm(residuals, axis=0)
        if np.all(residual_norms < threshold):
            break
        
        # Expand the subspace with the residuals
        for i in range(num_eigenvalues):
            if residual_norms[i] >= threshold:
                correction_vector = residuals[:, i] / (current_eigenvalues[i] - np.diag(matrixA))
                V = np.column_stack((V, correction_vector))
        
        # Re-orthonormalize the subspace
        V, _ = np.linalg.qr(V)
        
        # Limit the size of the subspace
        if V.shape[1] > subspace_dim:
            V = V[:, :subspace_dim]
    
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
