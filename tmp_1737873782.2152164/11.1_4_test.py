from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg



def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''

    
    # If dim is an integer, we are dealing with a single vector
    if isinstance(dim, int):
        out = np.zeros(dim)
        out[args] = 1.0
        return out

    # If dim is a list, create the tensor product of the basis vectors
    elif isinstance(dim, list) and isinstance(args, list) and len(dim) == len(args):
        basis_vectors = []
        for d, j in zip(dim, args):
            vector = np.zeros(d)
            vector[j] = 1.0
            basis_vectors.append(vector)

        # Compute the tensor product of all basis vectors
        out = basis_vectors[0]
        for v in basis_vectors[1:]:
            out = np.kron(out, v)
        return out

    # If input is invalid, return None
    else:
        return None


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