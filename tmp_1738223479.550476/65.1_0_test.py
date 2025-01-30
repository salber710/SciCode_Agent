from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.linalg import sqrtm
import itertools



# Background: The tensor product, also known as the Kronecker product, is an operation on two matrices of arbitrary size resulting in a block matrix. 
# If A is an m x n matrix and B is a p x q matrix, their Kronecker product A ⊗ B is an (m*p) x (n*q) matrix. 
# This operation is useful in quantum computing, where it describes the combined state space of two systems, and in various numerical methods and machine learning tasks.



def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one input is required")

    # Initialize the result with the first argument
    M = args[0]
    
    # Iterate over the rest of the arguments and compute the Kronecker product
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M


try:
    targets = process_hdf5_to_tuple('65.1', 3)
    target = targets[0]
    assert np.allclose(tensor([0,1],[0,1]), target)

    target = targets[1]
    assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)

    target = targets[2]
    assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e