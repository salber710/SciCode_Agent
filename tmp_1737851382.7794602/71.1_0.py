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
        j = args[0]
        out = np.zeros(dim)
        out[j] = 1.0
    elif isinstance(dim, list) and isinstance(args[0], list):
        # Multiple dimensions and multiple indices
        dims = dim
        indices = args[0]
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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.1', 3)
target = targets[0]

assert np.allclose(ket(2, 0), target)
target = targets[1]

assert np.allclose(ket(2, [1,1]), target)
target = targets[2]

assert np.allclose(ket([2,3], [0,1]), target)
