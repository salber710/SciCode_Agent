import numpy as np
from scipy.linalg import sqrtm
import itertools



# Background: The tensor product, also known as the Kronecker product, is an operation on two matrices of arbitrary size resulting in a block matrix. 
# It is a generalization of the outer product from vectors to matrices. The tensor product of two matrices A (of size m x n) and B (of size p x q) 
# is a matrix of size (m*p) x (n*q). The elements of the resulting matrix are computed as the product of each element of A with the entire matrix B. 
# This operation is associative, meaning the order of operations does not affect the final result, which allows us to extend it to an arbitrary number 
# of matrices or vectors. In the context of quantum mechanics and other fields, the tensor product is used to describe the state space of a composite 
# system as the product of the state spaces of its components.


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
    
    # Iterate over the remaining matrices/vectors and compute the tensor product
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
