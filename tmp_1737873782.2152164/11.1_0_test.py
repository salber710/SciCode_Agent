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

    
    if isinstance(dim, int):
        # If dim is an int, then args should be a list of indices for which we need to create tensor product
        if isinstance(args, list):
            # Create a single ket for each index in args and take the tensor product
            kets = [np.eye(dim)[:, j] for j in args]
            out = kets[0]
            for ket in kets[1:]:
                out = np.kron(out, ket)
            return out
        else:
            # If args is a single int, create a single basis vector
            out = np.zeros(dim)
            out[args] = 1
            return out
    
    elif isinstance(dim, list) and isinstance(args, list) and len(dim) == len(args):
        # If both dim and args are lists of the same length, create tensor products of basis vectors
        kets = [np.eye(d)[:, j] for d, j in zip(dim, args)]
        out = kets[0]
        for ket in kets[1:]:
            out = np.kron(out, ket)
        return out
    
    else:
        raise ValueError("Incompatible types or sizes of dim and args")


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