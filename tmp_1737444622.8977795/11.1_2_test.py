import numpy as np
import itertools
import scipy.linalg



# Background: 
# In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is represented as a column vector 
# with all elements being zero except for the j-th position, which is 1. This is known as a standard 
# basis vector. When dealing with tensor products of basis vectors, the resulting vector is constructed 
# by taking the Kronecker product. If j is a list and d is an integer, each element j_i of j specifies 
# which basis vector to take in a d-dimensional space, and the tensor product of these vectors is computed.
# If d is also a list, each d_i specifies the dimension of the space for the corresponding j_i, and 
# the tensor product of these basis vectors across different dimensions is computed.

def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''


    if isinstance(dim, int):
        # Single dimension case
        out = np.zeros(dim)
        out[args] = 1.0
    elif isinstance(dim, list):
        # Multi-dimensional case
        vectors = []
        for d, j in zip(dim, args):
            vec = np.zeros(d)
            vec[j] = 1.0
            vectors.append(vec)
        out = vectors[0]
        for vec in vectors[1:]:
            out = np.kron(out, vec)
    
    return out

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('11.1', 3)
target = targets[0]

assert np.allclose(ket(2, 0), target)
target = targets[1]

assert np.allclose(ket(2, [1,1]), target)
target = targets[2]

assert np.allclose(ket([2,3], [0,1]), target)
