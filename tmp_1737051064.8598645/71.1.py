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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('71.1', 3)
target = targets[0]

assert np.allclose(ket(2, 0), target)
target = targets[1]

assert np.allclose(ket(2, [1,1]), target)
target = targets[2]

assert np.allclose(ket([2,3], [0,1]), target)
