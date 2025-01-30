from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg



# Background: In quantum mechanics, a ket |j⟩ represents a column vector in a complex vector space, often used to describe quantum states. 
# A standard basis vector in a d-dimensional space is a vector with a 1 in the j-th position and 0s elsewhere. 
# The tensor product of vectors is a way to construct a new vector space from two or more vector spaces, 
# and it is used to describe the combined state of multiple quantum systems. 
# If j is a list, we need to construct the tensor product of the basis vectors for each j_i in the corresponding d_i-dimensional space.

def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''


    if isinstance(dim, int):
        # Single dimension case
        d = dim
        j = args
        # Create a d-dimensional basis vector with 1 at position j
        out = np.zeros(d)
        out[j] = 1.0
    else:
        # Multiple dimensions case
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


try:
    targets = process_hdf5_to_tuple('11.1', 3)
    target = targets[0]
    assert np.allclose(ket(2, 0), target)

    target = targets[1]
    assert np.allclose(ket(2, [1,1]), target)

    target = targets[2]
    assert np.allclose(ket([2,3], [0,1]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e