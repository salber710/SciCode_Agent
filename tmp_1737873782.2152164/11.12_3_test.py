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




def ket(dim, args):
    if isinstance(dim, int):
        out = np.zeros(dim)
        out[args] = 1.0
    elif isinstance(dim, list) and isinstance(args, list) and len(dim) == len(args):
        kets = []
        for d, j in zip(dim, args):
            ket = np.zeros(d)
            ket[j] = 1.0
            kets.append(ket)
        out = kets[0]
        for ket in kets[1:]:
            out = np.kron(out, ket)
    else:
        raise ValueError("Invalid input: dim and args should be lists of the same length or dim should be an int with args as int.")
    return out

def tensor(*args):
    if not args:
        raise ValueError("At least one matrix/vector must be provided for the tensor product.")
    M = args[0]
    for array in args[1:]:
        M = np.kron(M, array)
    return M

def generalized_amplitude_damping_channel(gamma, N):
    A1 = np.sqrt(gamma) * np.array([[1, 0], [0, np.sqrt(1 - N)]], dtype=np.float64)
    A2 = np.sqrt(gamma) * np.array([[0, np.sqrt(N)], [0, 0]], dtype=np.float64)
    A3 = np.sqrt(1 - gamma) * np.array([[np.sqrt(1 - N), 0], [0, 1]], dtype=np.float64)
    A4 = np.sqrt(1 - gamma) * np.array([[0, 0], [np.sqrt(N), 0]], dtype=np.float64)
    kraus = [A1, A2, A3, A4]
    return kraus

def apply_channel(K, rho, sys=None, dim=None):
    if sys is None and dim is None:
        new_rho = np.zeros_like(rho)
        for k in K:
            new_rho += k @ rho @ k.conj().T
        return new_rho
    elif sys is not None and dim is not None:
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
            ket_rest = tensor(*[np.eye(d)[idx] for d, idx in zip(dim, rest_indices)])
            bra_rest = ket_rest.conj().T
            projector_rest = np.outer(ket_rest, bra_rest)
            projector = tensor(projector_rest, projector_sys)
            sub_rho = projector @ rho @ projector
            for k in K:
                new_rho += projector @ (k @ sub_rho @ k.conj().T) @ projector
        return new_rho
    else:
        raise ValueError("Both sys and dim should be either None or specified as lists.")

def output_state(rails, gamma_1, N_1, gamma_2, N_2):
    dim_single_rail = 2
    dim_total = dim_single_rail ** rails
    
    initial_state = np.zeros((dim_total, dim_total), dtype=np.float64)
    for i in range(dim_total):
        initial_state[i, i] = 1.0 / dim_total
    
    K1 = generalized_amplitude_damping_channel(gamma_1, N_1)
    K2 = generalized_amplitude_damping_channel(gamma_2, N_2)
    
    rho_after_receiver_1 = apply_channel(K1, initial_state)
    rho_after_receiver_2 = apply_channel(K2, rho_after_receiver_1)
    
    return rho_after_receiver_2




def measurement(rails):
    '''Returns the measurement projector
    Input:
    rails: int, number of rails
    Output:
    global_proj: ( 2**(2*rails), 2**(2*rails) ) dimensional array of floats
    '''
    # The dimension of the space for each rail is 2, hence 2**rails for both parties
    dim = 2 ** rails
    dim_total = 2 ** (2 * rails)

    # Initialize the global projector
    global_proj = np.zeros((dim_total, dim_total), dtype=float)

    # Iterate over all possible positions of the single 1 in the binary representation
    for position in range(dim):
        # Create the one-particle state |0...010...0⟩ with 1 at 'position'
        ket_single_particle = np.zeros(dim, dtype=float)
        ket_single_particle[position] = 1.0

        # Compute the projector for this one-particle state
        projector_single_particle = np.outer(ket_single_particle, ket_single_particle)

        # Tensor product the projector with itself for both subsystems
        tensor_proj = np.kron(projector_single_particle, projector_single_particle)

        # Add this to the global projector
        global_proj += tensor_proj

    return global_proj


def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    perm: list of int containing the desired order
    dim: list of int containing the dimensions of all subsystems.
    Output:
    Y: 2d array of floats with equal dimensions, the density matrix of the permuted state
    '''
    # Calculate the total dimension of the system
    total_dim = int(np.prod(dim))

    # Check if dimensions match
    if X.shape != (total_dim, total_dim):
        raise ValueError("The dimensions of the density matrix do not match the dimensions of the subsystems.")

    # Calculate the new dimensions order
    new_dim = [dim[i] for i in perm]

    # Calculate the shape for reshaping the density matrix
    X_reshaped_shape = dim + dim
    X_reshaped = X.reshape(X_reshaped_shape)

    # Calculate the permutation indices for reshaping
    perm_indices = perm + [i + len(dim) for i in perm]

    # Permute the system using axes permutation
    X_permuted = np.transpose(X_reshaped, axes=perm_indices)

    # Reshape back to a matrix form
    Y = X_permuted.reshape(total_dim, total_dim)

    return Y




def syspermute(X, perm, dim):
    total_dim = int(np.prod(dim))
    if X.shape != (total_dim, total_dim):
        raise ValueError("The dimensions of the density matrix do not match the dimensions of the subsystems.")
    new_dim = [dim[i] for i in perm]
    X_reshaped_shape = dim + dim
    X_reshaped = X.reshape(X_reshaped_shape)
    perm_indices = perm + [i + len(dim) for i in perm]
    X_permuted = np.transpose(X_reshaped, axes=perm_indices)
    Y = X_permuted.reshape(total_dim, total_dim)
    return Y

def partial_trace(X, sys, dim):
    total_dim = int(np.prod(dim))
    if X.shape != (total_dim, total_dim):
        raise ValueError("The dimensions of the density matrix do not match the dimensions of the subsystems.")

    # Subsystems to keep
    keep = [i for i in range(len(dim)) if i not in sys]
    
    # Permute the density matrix to bring the subsystems to be traced to the end
    perm = keep + sys
    X_permuted = syspermute(X, perm, dim)
    
    # Calculate new dimensions after permuting
    dim_keep = [dim[i] for i in keep]
    dim_trace = [dim[i] for i in sys]
    
    # Reshape the permuted matrix to separate subsystems to keep and trace
    reshaped_dims = dim_keep + dim_trace + dim_keep + dim_trace
    X_reshaped = X_permuted.reshape(reshaped_dims)
    
    # Trace out the subsystems by summing over the axes corresponding to dim_trace
    trace_axes = tuple(range(len(dim_keep), len(dim_keep) + len(dim_trace)))
    X_traced = np.trace(X_reshaped, axis1=len(dim_keep), axis2=len(dim_keep) + len(dim_trace))
    
    # Reshape the traced result back to the size of the remaining subsystems
    result_dims = int(np.prod(dim_keep))
    Y = X_traced.reshape(result_dims, result_dims)
    
    return Y


def entropy(rho):
    '''Inputs:
    rho: 2d array of floats with equal dimensions, the density matrix of the state
    Output:
    en: quantum (von Neumann) entropy of the state rho, float
    '''
    # Compute the eigenvalues of the density matrix
    eigenvalues = np.linalg.eigvalsh(rho)
    
    # Filter out zero eigenvalues to avoid log(0) issues
    non_zero_eigenvalues = eigenvalues[eigenvalues > 0]
    
    # Calculate the von Neumann entropy using base-2 logarithm
    en = -np.sum(non_zero_eigenvalues * np.log2(non_zero_eigenvalues))
    
    return en


def coherent_inf_state(rho_AB, dimA, dimB):
    '''Inputs:
    rho_AB: 2d array of floats with equal dimensions, the state we evaluate coherent information
    dimA: int, dimension of system A
    dimB: int, dimension of system B
    Output
    co_inf: float, the coherent information of the state rho_AB
    '''
    # Use the partial_trace function to trace out system B and get rho_A
    rho_A = partial_trace(rho_AB, sys=[1], dim=[dimA, dimB])

    # Calculate the von Neumann entropy of rho_A
    entropy_A = entropy(rho_A)

    # Calculate the von Neumann entropy of the combined state rho_AB
    entropy_AB = entropy(rho_AB)

    # Coherent information is defined as S(B) - S(AB) = -S(A) + S(AB)
    co_inf = -entropy_A + entropy_AB

    return co_inf



def rate(rails, gamma_1, N_1, gamma_2, N_2):
    '''Inputs:
    rails: int, number of rails
    gamma_1: float, damping parameter of the first channel
    N_1: float, thermal parameter of the first channel
    gamma_2: float, damping parameter of the second channel
    N_2: float, thermal parameter of the second channel
    Output: float, the achievable rate of our protocol
    '''

    # Calculate the dimension of the total system
    dim_single_rail = 2
    dim_total = dim_single_rail ** (2 * rails)
    
    # Create the initial maximally entangled state
    initial_state = np.zeros((dim_total, dim_total), dtype=np.float64)
    for i in range(dim_total):
        initial_state[i, i] = 1.0 / dim_total
    
    # Apply the generalized amplitude damping channels to the state
    rho_after_channels = output_state(rails, gamma_1, N_1, gamma_2, N_2)
    
    # Calculate the global projector for the measurement
    global_proj = measurement(rails)
    
    # Post-select the state after the measurement
    post_selected_state = global_proj @ rho_after_channels @ global_proj
    post_selected_state /= np.trace(post_selected_state)  # Normalize the state
    
    # Calculate the coherent information of the post-selected state
    dim_subsystem = dim_single_rail ** rails
    coherent_info = coherent_inf_state(post_selected_state, dim_subsystem, dim_subsystem)
    
    # Return the rate, which is the coherent information per channel use
    return coherent_info


try:
    targets = process_hdf5_to_tuple('11.12', 5)
    target = targets[0]
    assert np.allclose(rate(2,0.2,0.2,0.2,0.2), target)

    target = targets[1]
    assert np.allclose(rate(2,0.3,0.4,0.2,0.2), target)

    target = targets[2]
    assert np.allclose(rate(3,0.4,0.1,0.1,0.2), target)

    target = targets[3]
    assert np.allclose(rate(2,0,0,0,0), target)

    target = targets[4]
    assert np.allclose(rate(2,0.2,0,0.4,0), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e