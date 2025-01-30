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
    # If sys and dim are None, apply the channel to the full system
    if sys is None and dim is None:
        new_rho = np.zeros_like(rho)
        # Apply each Kraus operator to the full system
        for k in K:
            new_rho += k @ rho @ k.conj().T
        return new_rho

    # If sys is given, apply the channel to each specified subsystem
    elif sys is not None and dim is not None:
        # Calculate the total dimension of the system
        total_dim = int(np.prod(dim))
        
        # Check if the dimensions match
        if rho.shape != (total_dim, total_dim):
            raise ValueError("The dimensions of the density matrix do not match the dimensions of the subsystems.")
        
        # Initialize the new density matrix
        new_rho = np.zeros_like(rho)

        # Iterate over all configurations of the other subsystems
        for indices in itertools.product(*[range(d) for d in dim]):
            # Prepare the indices for the systems being acted upon
            indices_sys = tuple(indices[i] for i in sys)
            
            # Compute the tensor product of |indices_sys⟩⟨indices_sys|
            ket_indices_sys = np.zeros(dim[sys[0]])
            ket_indices_sys[indices_sys] = 1.0
            bra_indices_sys = ket_indices_sys.conj().T
            projector_sys = np.outer(ket_indices_sys, bra_indices_sys)
            
            # Compute the rest of the indices
            rest_indices = tuple(idx for i, idx in enumerate(indices) if i not in sys)

            # Calculate the rest of the state
            ket_rest = tensor(*[np.eye(d)[idx] for d, idx in zip(dim, rest_indices)])
            bra_rest = ket_rest.conj().T
            projector_rest = np.outer(ket_rest, bra_rest)

            # Construct the projector for the current configuration
            projector = tensor(projector_rest, projector_sys)

            # Extract the sub-matrix corresponding to the current configuration
            sub_rho = projector @ rho @ projector

            # Apply each Kraus operator to the sub-matrix and sum to new_rho
            for k in K:
                new_rho += projector @ (k @ sub_rho @ k.conj().T) @ projector

        return new_rho

    else:
        raise ValueError("Both sys and dim should be either None or specified as lists.")


def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''

    # Kraus operators for the generalized amplitude damping channel
    A1 = np.sqrt(gamma) * np.array([[1, 0], [0, np.sqrt(1 - N)]], dtype=np.float64)
    A2 = np.sqrt(gamma) * np.array([[0, np.sqrt(N)], [0, 0]], dtype=np.float64)
    A3 = np.sqrt(1 - gamma) * np.array([[np.sqrt(1 - N), 0], [0, 1]], dtype=np.float64)
    A4 = np.sqrt(1 - gamma) * np.array([[0, 0], [np.sqrt(N), 0]], dtype=np.float64)

    kraus = [A1, A2, A3, A4]

    return kraus



def output_state(rails, gamma_1, N_1, gamma_2, N_2):
    '''Inputs:
    rails: int, number of rails
    gamma_1: float, damping parameter of the first channel
    N_1: float, thermal parameter of the first channel
    gamma_2: float, damping parameter of the second channel
    N_2: float, thermal parameter of the second channel
    Output
    state: 2**(2*rails) x 2**(2*rails) dimensional array of floats, the output state
    '''

    # Import necessary dependencies



    # Step 1: Generate the initial maximally entangled state for m-rails
    dim_per_rail = 2  # Each rail can be in state |0⟩ or |1⟩
    dim_total = dim_per_rail ** rails  # Total dimension for one party
    total_dim = dim_total ** 2  # Total dimension for the bipartite system

    # Create the initial entangled state |Φ+⟩ = 1/sqrt(dim_total) * sum_i |i⟩|i⟩
    entangled_state = np.zeros((total_dim, total_dim), dtype=np.float64)
    for i in range(dim_total):
        ket_i = np.zeros(dim_total)
        ket_i[i] = 1
        outer_product = np.outer(ket_i, ket_i)
        entangled_state += np.kron(outer_product, outer_product)
    
    entangled_state /= dim_total

    # Step 2: Get Kraus operators for the generalized amplitude damping channels
    def generalized_amplitude_damping_channel(gamma, N):
        '''Generates the generalized amplitude damping channel.'''
        A1 = np.sqrt(gamma) * np.array([[1, 0], [0, np.sqrt(1 - N)]], dtype=np.float64)
        A2 = np.sqrt(gamma) * np.array([[0, np.sqrt(N)], [0, 0]], dtype=np.float64)
        A3 = np.sqrt(1 - gamma) * np.array([[np.sqrt(1 - N), 0], [0, 1]], dtype=np.float64)
        A4 = np.sqrt(1 - gamma) * np.array([[0, 0], [np.sqrt(N), 0]], dtype=np.float64)
        return [A1, A2, A3, A4]

    # Step 3: Apply the Kraus operators to the entangled state
    def apply_channel(K, rho, sys, dim):
        '''Applies the channel with Kraus operators in K to the state rho on specified subsystems.'''
        total_dim = int(np.prod(dim))
        if rho.shape != (total_dim, total_dim):
            raise ValueError("The dimensions of the density matrix do not match the dimensions of the subsystems.")
        
        new_rho = np.zeros_like(rho)
        for indices in itertools.product(*[range(d) for d in dim]):
            indices_sys = tuple(indices[i] for i in sys)
            ket_indices_sys = np.zeros(dim[sys[0]])
            ket_indices_sys[indices_sys] = 1.0
            bra_indices_sys = ket_indices_sys.conj().T
            projector_sys = np.outer(ket_indices_sys, bra_indices_sys)

            rest_indices = tuple(idx for i, idx in enumerate(indices) if i not in sys)
            ket_rest = np.eye(dim_total)[rest_indices[0]]
            bra_rest = ket_rest.conj().T
            projector_rest = np.outer(ket_rest, bra_rest)

            projector = np.kron(projector_rest, projector_sys)
            sub_rho = projector @ rho @ projector

            for k in K:
                new_rho += projector @ (k @ sub_rho @ k.conj().T) @ projector

        return new_rho

    # Define dimensions and systems for the channels
    sys_1 = list(range(rails))  # First m subsystems for receiver 1
    sys_2 = list(range(rails, 2*rails))  # Next m subsystems for receiver 2
    dim = [2] * (2 * rails)  # Dimension for each subsystem

    # Get Kraus operators for each channel
    K1 = generalized_amplitude_damping_channel(gamma_1, N_1)
    K2 = generalized_amplitude_damping_channel(gamma_2, N_2)

    # Apply channels
    output_state_1 = apply_channel(K1, entangled_state, sys_1, dim)
    output_state_2 = apply_channel(K2, output_state_1, sys_2, dim)

    return output_state_2


try:
    targets = process_hdf5_to_tuple('11.6', 3)
    target = targets[0]
    assert np.allclose(output_state(2,0,0,0,0), target)

    target = targets[1]
    assert np.allclose(output_state(2,1,0,1,0), target)

    target = targets[2]
    assert np.allclose(output_state(2,1,1,1,1), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e