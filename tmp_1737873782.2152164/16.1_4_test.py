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
    # Start by creating an empty matrix of the given dimension
    A = np.zeros((dim, dim))
    
    # Fill the diagonal with increasing integer values starting from 1
    for i in range(dim):
        A[i, i] = i + 1
    
    # Add random noise to the matrix
    random_noise = noise * np.random.randn(dim, dim)
    
    # Adding the symmetric noise to the matrix
    A = A + random_noise
    
    # Symmetrize the matrix by averaging with its transpose
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