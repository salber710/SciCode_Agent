import numpy as np
import itertools
import scipy.linalg



# Background: In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is a column vector with a 1 in the j-th position and 0s elsewhere. 
# This is known as a standard basis vector. When dealing with multiple quantum systems, the state of the combined system is represented by the 
# tensor product of the individual states. If j is a list, it represents multiple indices for which we need to create a tensor product of 
# standard basis vectors. If d is a list, it specifies the dimensions of each individual space for the tensor product.




def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single space case
        d = dim
        j = args
        # Create a d-dimensional basis vector with 1 at position j
        out = np.zeros(d)
        out[j] = 1.0
    else:
        # Multiple spaces case
        d_list = dim
        j_list = args
        # Create the tensor product of basis vectors
        basis_vectors = []
        for d, j in zip(d_list, j_list):
            vec = np.zeros(d)
            vec[j] = 1.0
            basis_vectors.append(vec)
        # Compute the tensor product
        out = basis_vectors[0]
        for vec in basis_vectors[1:]:
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
