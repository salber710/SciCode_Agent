import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

# Background: In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is a column vector with a 1 in the j-th position and 0s elsewhere. 
# This is a standard basis vector in the context of quantum states. When dealing with multiple quantum systems, the tensor product of individual 
# kets is used to represent the combined state. The tensor product of vectors is a way to construct a new vector space from two or more vector spaces. 
# If j is a list, it represents multiple indices for which we need to create a tensor product of basis vectors. Similarly, if d is a list, it 
# represents the dimensions of each corresponding basis vector.


def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single dimension and single index
        out = np.zeros(dim)
        out[args] = 1.0
    else:
        # Multiple dimensions and indices
        vectors = []
        for d, j in zip(dim, args):
            vec = np.zeros(d)
            vec[j] = 1.0
            vectors.append(vec)
        # Compute the tensor product of all vectors
        out = vectors[0]
        for vec in vectors[1:]:
            out = np.kron(out, vec)
    
    return out



# Background: In linear algebra, the tensor product (also known as the Kronecker product) of two matrices is a way to construct a new matrix from two given matrices. 
# The tensor product of matrices is a generalization of the outer product of vectors. If A is an m×n matrix and B is a p×q matrix, 
# then their tensor product A ⊗ B is an mp×nq matrix. This operation is widely used in quantum mechanics to describe the state of a composite quantum system. 
# The tensor product of multiple matrices is computed by iteratively applying the Kronecker product to pairs of matrices.

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''

    
    if not args:
        raise ValueError("At least one matrix/vector is required for the tensor product.")
    
    # Start with the first matrix/vector
    M = args[0]
    
    # Iteratively compute the tensor product with the remaining matrices/vectors
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
