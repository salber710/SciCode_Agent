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
    # Import the necessary modules




    # Function to generate the tensor product state |0L> = |01> - |10> for 1 rail
    def logical_zero_state():
        # |01> = [0, 1, 0, 0], |10> = [0, 0, 1, 0]
        ket_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
        ket_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
        return (ket_01 - ket_10) / np.sqrt(2)

    # Function to generate the tensor product state |1L> = |01> + |10> for 1 rail
    def logical_one_state():
        # |01> = [0, 1, 0, 0], |10> = [0, 0, 1, 0]
        ket_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
        ket_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
        return (ket_01 + ket_10) / np.sqrt(2)

    # Generate the logical zero and one for the given number of rails
    logical_zero = logical_zero_state()
    logical_one = logical_one_state()

    # Generate the full bipartite entangled state in the form (|0L>|0L> + |1L>|1L>)/sqrt(2)
    entangled_state = np.kron(logical_zero, logical_zero) + np.kron(logical_one, logical_one)
    entangled_state /= np.sqrt(2)  # Normalize the state

    # Create the density matrix from the entangled state
    density_matrix = np.outer(entangled_state, entangled_state.conj())

    return density_matrix


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