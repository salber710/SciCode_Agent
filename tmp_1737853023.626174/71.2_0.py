import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

# Background: In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is a column vector with a 1 in the j-th position and 0s elsewhere. 
# This is a standard basis vector in the context of quantum states. When dealing with multiple quantum systems, the tensor product of individual 
# kets is used to represent the combined state. The tensor product of vectors results in a higher-dimensional vector space, where the dimensions 
# are the product of the individual dimensions. In this problem, we need to construct such a ket vector or a tensor product of multiple ket vectors 
# based on the input dimensions and indices.


def ket(dim, *args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single dimension and single index
        if not isinstance(args[0], int):
            raise TypeError("For single dimension, index must be an integer.")
        j = args[0]
        if j < 0 or j >= dim:
            raise IndexError("Index out of bounds.")
        out = np.zeros(dim)
        out[j] = 1.0
    elif isinstance(dim, list) and isinstance(args[0], list):
        # Multiple dimensions and multiple indices
        dims = dim
        indices = args[0]
        if len(dims) != len(indices):
            raise ValueError("Dimensions and indices must have the same length.")
        if not dims or not indices:
            raise ValueError("Dimensions and indices cannot be empty.")
        # Validate dimensions are positive and indices are within bounds
        for d, idx in zip(dims, indices):
            if d <= 0:
                raise ValueError("Dimensions must be positive integers.")
            if idx < 0 or idx >= d:
                raise IndexError("Index out of bounds.")
        # Start with the first ket
        out = np.zeros(dims[0])
        out[indices[0]] = 1.0
        # Tensor product with subsequent kets
        for d, j in zip(dims[1:], indices[1:]):
            ket_j = np.zeros(d)
            ket_j[j] = 1.0
            out = np.kron(out, ket_j)
    else:
        raise ValueError("Invalid input format for dim and args.")
    
    return out



# Background: In linear algebra and quantum mechanics, the tensor product (also known as the Kronecker product) is an operation on two matrices or vectors that results in a block matrix. For vectors, the tensor product results in a higher-dimensional vector. For matrices, it results in a larger matrix that combines the information of the input matrices. The tensor product is essential in quantum mechanics for describing the state of a composite quantum system. The Kronecker product of matrices A (of size m x n) and B (of size p x q) is a matrix of size (m*p) x (n*q).


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one input matrix/vector is required.")
    
    # Start with the first matrix/vector
    M = args[0]
    
    # Iterate over the remaining matrices/vectors and compute the Kronecker product
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.2', 3)
target = targets[0]

assert np.allclose(tensor([0,1],[0,1]), target)
target = targets[1]

assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)
target = targets[2]

assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)
