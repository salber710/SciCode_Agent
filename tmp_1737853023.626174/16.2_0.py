import math
import numpy as np

# Background: A symmetric matrix is a square matrix that is equal to its transpose. 
# In this task, we need to create a symmetric matrix with increasing values along its diagonal. 
# The diagonal elements will be integers starting from 1 up to the dimension of the matrix. 
# Each element of the matrix will be modified by a product of a normally distributed random number 
# and a user-provided noise level. The symmetrization process involves averaging the matrix with its transpose 
# to ensure symmetry. The numpy library provides functions to generate random numbers and perform matrix operations.


def init_matrix(dim, noise):
    '''Generate a symmetric matrix with increasing values along its diagonal.
    Inputs:
    - dim: The dimension of the matrix (int).
    - noise: Noise level (float).
    Output:
    - A: a 2D array where each element is a float, representing the symmetric matrix.
    '''
    if dim < 0:
        raise ValueError("Dimension must be non-negative")
    
    # Initialize a matrix with zeros
    A = np.zeros((dim, dim))
    
    # Fill the diagonal with increasing values
    for i in range(dim):
        A[i, i] = i + 1
    
    # Modify each element by a product of a normally distributed random number and the noise level
    random_noise = np.random.normal(0, 1, (dim, dim))
    A += noise * random_noise
    
    # Symmetrize the matrix by averaging it with its transpose
    A = (A + A.T) / 2
    
    return A



# Background: Davidson's method is an iterative algorithm used to find a few of the lowest eigenvalues and corresponding eigenvectors of a large sparse symmetric matrix. 
# It is particularly useful in quantum chemistry and physics for solving large eigenvalue problems. The method starts with an initial guess for the eigenvectors and 
# iteratively refines them. The algorithm constructs a subspace and projects the matrix onto this subspace to find approximate eigenvalues and eigenvectors. 
# The convergence is controlled by a threshold, and the process continues until the change in eigenvalues is below this threshold.



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
    if num_eigenvalues > n:
        raise ValueError("Number of eigenvalues requested exceeds matrix dimension")

    # Initial guess: start with a random vector
    V = np.random.rand(n, num_eigenvalues)
    V, _ = np.linalg.qr(V)  # Orthonormalize the initial guess

    current_eigenvalues = np.zeros(num_eigenvalues)
    prev_eigenvalues = np.ones(num_eigenvalues) * np.inf

    while np.max(np.abs(current_eigenvalues - prev_eigenvalues)) > threshold:
        # Project matrixA onto the subspace spanned by V
        AV = np.dot(matrixA, V)
        T = np.dot(V.T, AV)

        # Solve the small eigenvalue problem
        eigvals, eigvecs = np.linalg.eigh(T)

        # Update eigenvalues
        prev_eigenvalues = current_eigenvalues
        current_eigenvalues = eigvals[:num_eigenvalues]

        # Compute the residuals
        residuals = AV - np.dot(V, np.dot(T, eigvecs))

        # Update the subspace
        for i in range(num_eigenvalues):
            r = residuals[:, i]
            if np.linalg.norm(r) > threshold:
                # Normalize the residual
                r /= np.linalg.norm(r)
                # Orthogonalize against the current subspace
                for j in range(V.shape[1]):
                    r -= np.dot(V[:, j], r) * V[:, j]
                # Add the new vector to the subspace
                V = np.column_stack((V, r))
                # Re-orthonormalize the subspace
                V, _ = np.linalg.qr(V)

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
