import numpy as np
from scipy.linalg import sqrtm
import itertools



# Background: The tensor product, also known as the Kronecker product, is an operation on two matrices (or vectors) that results in a block matrix. 
# If A is an m x n matrix and B is a p x q matrix, their Kronecker product A ⊗ B is an mp x nq matrix. 
# The Kronecker product is a generalization of the outer product from vectors to matrices. 
# It is used in various fields such as quantum computing, signal processing, and the study of multi-linear algebra. 
# In this function, we aim to compute the tensor product of an arbitrary number of matrices or vectors.


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one matrix or vector is required for the tensor product.")
    
    # Start with the first matrix/vector
    M = args[0]
    
    # Iterate over the remaining matrices/vectors and compute the Kronecker product
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('65.1', 3)
target = targets[0]

assert np.allclose(tensor([0,1],[0,1]), target)
target = targets[1]

assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)
target = targets[2]

assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)
