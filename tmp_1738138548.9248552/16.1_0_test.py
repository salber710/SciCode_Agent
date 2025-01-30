from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import math
import numpy as np



# Background: A symmetric matrix is a square matrix that is equal to its transpose. 
# In this task, we need to create a symmetric matrix with increasing values along its diagonal. 
# The diagonal elements will be integers starting from 1 up to the dimension of the matrix. 
# Each element of the matrix will be modified by a product of a normally distributed random number 
# and a user-provided noise level. The matrix is then symmetrized by averaging it with its transpose. 
# This ensures that the resulting matrix is symmetric, as the average of a matrix and its transpose 
# is always symmetric.


def init_matrix(dim, noise):
    '''Generate a symmetric matrix with increasing values along its diagonal.
    Inputs:
    - dim: The dimension of the matrix (int).
    - noise: Noise level (float).
    Output:
    - A: a 2D array where each element is a float, representing the symmetric matrix.
    '''
    # Initialize a matrix with zeros
    A = np.zeros((dim, dim))
    
    # Fill the diagonal with increasing values
    for i in range(dim):
        A[i, i] = i + 1
    
    # Modify each element by a product of a normally distributed random number and the noise level
    random_noise = np.random.normal(0, 1, (dim, dim))
    A = A + noise * random_noise
    
    # Symmetrize the matrix by averaging it with its transpose
    A = (A + A.T) / 2
    
    return A


try:
    targets = process_hdf5_to_tuple('16.1', 3)
    target = targets[0]
    np.random.seed(1000)
    assert np.allclose(init_matrix(10,0.), target)

    target = targets[1]
    np.random.seed(1000)
    assert np.allclose(init_matrix(5,0.1), target)

    target = targets[2]
    np.random.seed(1000)
    assert np.allclose(init_matrix(1000,0.00001), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e