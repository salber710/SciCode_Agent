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
    # Calculate the dimension of the Hilbert space for each party
    dim_per_rail = 2  # Since each rail can be in state |0⟩ or |1⟩
    dim_total = dim_per_rail ** rails  # Total dimension for each party

    # Create the maximally entangled state
    entangled_state = np.zeros((dim_total, dim_total), dtype=np.float64)
    
    # Iterate over all basis states for one party
    for basis_index in range(dim_total):
        # Prepare the basis vector |i⟩ for the first party
        ket_i = ket(dim_total, basis_index)

        # Prepare the basis vector |i⟩ for the second party
        ket_j = ket(dim_total, basis_index)

        # Outer product |i⟩⟨j| to build the density matrix
        outer_product = np.outer(ket_i, ket_j)

        # Sum over all outer products to create the maximally entangled state
        entangled_state += outer_product

    # Normalize the state
    entangled_state /= dim_total

    return entangled_state


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one matrix/vector must be provided for the tensor product.")

    M = args[0]
    for array in args[1:]:
        M = np.kron(M, array)

    return M



def apply_channel(K, rho, sys=None, dim=None):
    '''Applies the channel with Kraus operators in K to the state rho on
    systems specified by the list sys. The dimensions of the subsystems of
    rho are given by dim.
    Inputs:
    K: list of 2d array of floats, list of Kraus operators
    rho: 2d array of floats, input density matrix
    sys: list of int or None, list of subsystems to apply the channel, None means full system
    dim: list of int or None, list of dimensions of each subsystem, None means full system
    Output:
    matrix: output density matrix of floats
    '''
    if sys is None and dim is None:
        dim_total = rho.shape[0]
        sys_dim = dim_total
        sys_ids = [0]
    elif sys is not None and dim is not None:
        sys_ids = sys
        sys_dim = np.prod([dim[i] for i in sys])
        dim_total = int(rho.shape[0] / sys_dim)
    else:
        raise ValueError("Both sys and dim should be either None or defined.")

    dim_rest = int(dim_total / sys_dim)
    
    # Create new density matrix with channel applied
    new_rho = np.zeros_like(rho, dtype=np.complex128)
    
    for k in K:
        # Expand the Kraus operator to the full space if necessary
        if sys is not None:
            k_full = np.eye(dim_total, dtype=np.complex128)
            indices = [slice(None)] * len(k_full.shape)
            indices[sys_ids[0]:sys_ids[0] + len(k.shape)] = [slice(None) for _ in k.shape]
            k_full[tuple(indices)] = k
        else:
            k_full = k
        
        # Apply the channel using the Kraus operators
        new_rho += k_full @ rho @ k_full.conj().T

    return new_rho


try:
    targets = process_hdf5_to_tuple('11.4', 3)
    target = targets[0]
    K = [np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]])]
    rho = np.ones((2,2))/2
    assert np.allclose(apply_channel(K, rho, sys=None, dim=None), target)

    target = targets[1]
    K = [np.sqrt(0.8)*np.eye(2),np.sqrt(0.2)*np.array([[0,1],[1,0]])]
    rho = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    assert np.allclose(apply_channel(K, rho, sys=[2], dim=[2,2]), target)

    target = targets[2]
    K = [np.sqrt(0.8)*np.eye(2),np.sqrt(0.2)*np.array([[0,1],[1,0]])]
    rho = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    assert np.allclose(apply_channel(K, rho, sys=[1,2], dim=[2,2]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e