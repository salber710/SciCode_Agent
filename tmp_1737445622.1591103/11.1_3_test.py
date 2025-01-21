from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg



# Background: In quantum mechanics, a ket is a vector in a complex vector space, often used to represent quantum states. 
# A standard basis vector in a d-dimensional space is a vector where all elements are zero except for the j-th element, which is one.
# The tensor product is an operation that takes two vectors and returns a new vector in a higher-dimensional space, 
# representing the combined state of two systems. If j is a list of indices and d is a list of dimensions, 
# we compute the tensor product of the individual basis vectors corresponding to each index in their respective dimensions.

def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''

    
    if isinstance(dim, int):
        # dim is a single integer, create a single standard basis vector
        out = np.zeros(dim)
        out[args] = 1.0
    elif isinstance(dim, list):
        # dim is a list, create a tensor product of basis vectors
        vectors = []
        for d, j in zip(dim, args):
            vec = np.zeros(d)
            vec[j] = 1.0
            vectors.append(vec)
        
        # Compute the tensor product of the vectors
        out = vectors[0]
        for vec in vectors[1:]:
            out = np.kron(out, vec)
    
    return out


try:
    targets = process_hdf5_to_tuple('11.1', 3)
    target = targets[0]
    assert np.allclose(ket(2, 0), target)

    target = targets[1]
    assert np.allclose(ket(2, [1,1]), target)

    target = targets[2]
    assert np.allclose(ket([2,3], [0,1]), target)

