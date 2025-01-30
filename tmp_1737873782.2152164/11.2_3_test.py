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
        # Single dimension basis vector
        out = np.zeros(dim)
        out[args] = 1.0
    elif isinstance(dim, list) and isinstance(args, list) and len(dim) == len(args):
        # Tensor product case
        # Compute the individual kets
        kets = []
        for d, j in zip(dim, args):
            ket = np.zeros(d)
            ket[j] = 1.0
            kets.append(ket)
        
        # Compute the tensor product of all kets
        out = kets[0]
        for ket in kets[1:]:
            out = np.kron(out, ket)
    else:
        raise ValueError("Invalid input: dim and args should be lists of the same length or dim should be an int with args as int.")

    return out



def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    # Dimension of each rail is 2 (|0> and |1>)
    dim = 2
    total_dim = 2 ** (2 * rails)

    # Create a zero density matrix
    state = np.zeros((total_dim, total_dim), dtype=np.float64)

    # Iterate over each possible rail encoding
    for j in range(2 ** rails):
        # Encode |j>|j> using the basis vector function
        basis_vector = ket(dim=[2] * (2 * rails), args=[int(x) for x in f"{j:0{rails}b}" * 2])
        
        # Add the outer product to the density matrix
        state += np.outer(basis_vector, basis_vector)

    # Normalize the density matrix
    state /= (2 ** rails)

    return state


try:
    targets = process_hdf5_to_tuple('11.2', 3)
    target = targets[0]
    assert np.allclose(multi_rail_encoding_state(1), target)

    target = targets[1]
    assert np.allclose(multi_rail_encoding_state(2), target)

    target = targets[2]
    assert np.allclose(multi_rail_encoding_state(3), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e