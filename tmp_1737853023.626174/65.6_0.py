import numpy as np
from scipy.linalg import sqrtm
import itertools

# Background: The tensor product, also known as the Kronecker product, is an operation on two matrices of arbitrary size resulting in a block matrix. 
# It is a generalization of the outer product from vectors to matrices. The tensor product of two matrices A (of size m x n) and B (of size p x q) 
# is a matrix of size (m*p) x (n*q). The elements of the resulting matrix are computed as the product of each element of A with the entire matrix B. 
# This operation is associative, meaning the order of operations does not affect the final result, which allows us to extend it to an arbitrary number 
# of matrices or vectors. In the context of quantum mechanics and other fields, the tensor product is used to describe the state space of a composite 
# system as the product of the state spaces of its components.


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one matrix or vector is required for the tensor product.")
    
    # Start with the first matrix/vector
    M = args[0]
    
    # Iterate over the remaining matrices/vectors and compute the tensor product
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M


# Background: In quantum mechanics, a quantum channel is a mathematical model for the physical process of transmitting quantum states. 
# It is represented by a set of Kraus operators, which are matrices that describe the effect of the channel on a quantum state. 
# The action of a quantum channel on a density matrix (quantum state) is given by the sum of transformations of the state by each Kraus operator. 
# Specifically, if K = {K1, K2, ..., Kn} is a set of Kraus operators, the channel transforms a density matrix ρ as: 
# ρ' = Σ_i (Ki * ρ * Ki†), where Ki† is the conjugate transpose of Ki. 
# When applying a channel to specific subsystems of a composite quantum system, we need to consider the tensor product structure of the state. 
# The function should apply the channel to specified subsystems, which requires reshaping and permuting the state according to the dimensions of the subsystems.



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
    if sys is None or dim is None:
        # Apply the channel to the entire system
        new_rho = np.zeros_like(rho, dtype=np.complex128)
        for kraus in K:
            new_rho += kraus @ rho @ kraus.conj().T
        return new_rho
    else:
        # Apply the channel to specified subsystems
        total_dim = np.prod(dim)
        if rho.shape != (total_dim, total_dim):
            raise ValueError("The dimensions of rho do not match the product of the subsystem dimensions.")
        
        # Reshape rho into a multi-dimensional array
        rho_reshaped = rho.reshape(tuple(dim) * 2)
        
        # Prepare the identity for unaffected subsystems
        unaffected_dims = [dim[i] for i in range(len(dim)) if i not in sys]
        identity = np.eye(int(np.prod(unaffected_dims)), dtype=np.complex128)
        
        # Initialize the new rho
        new_rho = np.zeros_like(rho_reshaped, dtype=np.complex128)
        
        # Iterate over all combinations of indices for the unaffected subsystems
        for indices in itertools.product(*[range(d) for d in unaffected_dims]):
            # Construct the slice for the current indices
            slice_indices = tuple(slice(None) if i in sys else indices[unaffected_dims.index(dim[i])] for i in range(len(dim)))
            
            # Extract the relevant part of rho
            rho_part = rho_reshaped[slice_indices]
            
            # Apply the channel to this part
            new_rho_part = np.zeros_like(rho_part, dtype=np.complex128)
            for kraus in K:
                new_rho_part += kraus @ rho_part @ kraus.conj().T
            
            # Place the transformed part back into the new rho
            new_rho[slice_indices] = new_rho_part
        
        # Reshape back to the original 2D form
        return new_rho.reshape(total_dim, total_dim)


# Background: In quantum information theory, a qubit is the basic unit of quantum information, analogous to a bit in classical information theory. 
# A 2n-qubit system is a composite quantum system consisting of 2n qubits. Quantum channels are used to model noise and other transformations 
# that occur to quantum states during transmission or processing. When dealing with composite systems, it is common to apply different quantum 
# channels to different parts of the system. In this problem, we have a 2n-qubit input state, and we want to apply one quantum channel (channel1) 
# to the first n qubits and another quantum channel (channel2) to the last n qubits. If channel2 is not provided, it defaults to channel1. 
# The function apply_channel is used to apply the Kraus operators of a quantum channel to specific subsystems of a quantum state.



def apply_channel(kraus_ops, rho, sys, dim):
    '''Applies a quantum channel, described by Kraus operators, to part of a quantum state.
    
    Args:
        kraus_ops (list of np.ndarray): The Kraus operators for the channel.
        rho (np.ndarray): The density matrix of the quantum state.
        sys (list): The indices of the qubits the channel is applied to.
        dim (list): The dimensions of the Hilbert spaces for each qubit.
    
    Returns:
        np.ndarray: The resulting quantum state after applying the channel.
    '''
    subsystem_dims = [dim[i] for i in sys]
    other_dims = [dim[i] for i in range(len(dim)) if i not in sys]
    
    # Reshape the state into a tensor with separate indices for each subsystem
    tensor_shape = other_dims + subsystem_dims
    state_tensor = rho.reshape(tensor_shape)
    
    # Permute to move the subsystem indices to the front
    perm = list(range(len(other_dims), len(dim))) + list(range(len(other_dims)))
    state_tensor = np.transpose(state_tensor, perm)
    
    # Apply the channel to the subsystem by summing over the Kraus operators
    new_state_tensor = np.zeros_like(state_tensor)
    for K in kraus_ops:
        K_tensor = K
        for _ in range(len(other_dims)):
            K_tensor = np.expand_dims(K_tensor, axis=0)
        new_state_tensor += K_tensor @ state_tensor @ K_tensor.conj().T
    
    # Permute back and reshape
    inv_perm = np.argsort(perm)
    new_state_tensor = np.transpose(new_state_tensor, inv_perm)
    new_state = new_state_tensor.reshape(rho.shape)
    
    return new_state

def channel_output(input_state, channel1, channel2=None):
    '''Returns the channel output
    Inputs:
        input_state: density matrix of the input 2n qubit state, ( 2**(2n), 2**(2n) ) array of floats
        channel1: kraus operators of the first channel, list of (2,2) array of floats
        channel2: kraus operators of the second channel, list of (2,2) array of floats
    Output:
        output: the channel output, ( 2**(2n), 2**(2n) ) array of floats
    '''
    # Determine the number of qubits n
    total_dim = input_state.shape[0]
    n = int(np.log2(total_dim) / 2)
    
    # If channel2 is None, set it to channel1
    if channel2 is None:
        channel2 = channel1
    
    # Dimensions of each qubit
    dim = [2] * (2 * n)
    
    # Apply channel1 to the first n qubits
    output_state = apply_channel(channel1, input_state, sys=list(range(n)), dim=dim)
    
    # Apply channel2 to the last n qubits
    output_state = apply_channel(channel2, output_state, sys=list(range(n, 2 * n)), dim=dim)
    
    return output_state


# Background: In quantum mechanics, a parity measurement is a type of measurement that checks whether the number of qubits in the state |1⟩ is even or odd. 
# For a set of n qubits, an even parity measurement means that the number of qubits in the state |1⟩ is even (including zero), while an odd parity measurement 
# means that the number is odd. In this protocol, we perform parity measurements on two sets of n qubits. If both sets have even parity, the state is kept and 
# transformed into a two-qubit state. Specifically, the state |00...0⟩ (n zeros) is transformed into |0⟩, and the state |11...1⟩ (n ones) is transformed into |1⟩. 
# If either set has odd parity, the state is discarded. This process is a form of post-selection, where only certain measurement outcomes are kept.



def ghz_protocol(state):
    '''Returns the output state of the protocol
    Input:
    state: 2n qubit input state, 2^2n by 2^2n array of floats, where n is determined from the size of the input state
    Output:
    post_selected: the output state
    '''
    # Determine the number of qubits n
    total_dim = state.shape[0]
    n = int(np.log2(total_dim) / 2)
    
    # Validate input dimensions
    if total_dim != 2**(2*n):
        raise ValueError("Input state must be a 2^(2n) by 2^(2n) matrix.")
    
    # Initialize the post-selected state
    post_selected = np.zeros((2, 2), dtype=state.dtype)
    
    # Iterate over all possible basis states for the first n qubits
    for basis1 in itertools.product([0, 1], repeat=n):
        # Check if the first n qubits have even parity
        if sum(basis1) % 2 != 0:
            continue
        
        # Iterate over all possible basis states for the last n qubits
        for basis2 in itertools.product([0, 1], repeat=n):
            # Check if the last n qubits have even parity
            if sum(basis2) % 2 != 0:
                continue
            
            # Calculate the index in the original state
            index1 = int(''.join(map(str, basis1)), 2)
            index2 = int(''.join(map(str, basis2)), 2)
            
            # Map the basis states to the two-qubit state
            if basis1 == (0,) * n:
                new_index1 = 0
            elif basis1 == (1,) * n:
                new_index1 = 1
            else:
                continue
            
            if basis2 == (0,) * n:
                new_index2 = 0
            elif basis2 == (1,) * n:
                new_index2 = 1
            else:
                continue
            
            # Add the contribution to the post-selected state
            post_selected[new_index1, new_index2] += state[index1, index2]
    
    return post_selected


# Background: In quantum information theory, fidelity is a measure of the "closeness" or similarity between two quantum states. 
# It is particularly useful for comparing a theoretical state with an experimentally obtained state. For two density matrices ρ and σ, 
# the fidelity F(ρ, σ) is defined as F(ρ, σ) = (Tr(sqrt(sqrt(ρ) * σ * sqrt(ρ))))^2. This formula involves taking the square root of 
# the density matrices and computing the trace of the resulting matrix. Fidelity ranges from 0 to 1, where 1 indicates that the states 
# are identical, and 0 indicates that they are orthogonal.



def fidelity(rho, sigma):
    '''Returns the fidelity between two states.
    Inputs:
    rho, sigma: density matrices of the two states, 2d array of floats
    Output:
    fid: fidelity, float
    '''
    # Validate input matrices are square and Hermitian
    if rho is None or sigma is None:
        raise TypeError("Input matrices cannot be None.")
    if rho.shape[0] != rho.shape[1] or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("Density matrices must be square.")
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("Matrix rho is not Hermitian.")
    if not np.allclose(sigma, sigma.conj().T):
        raise ValueError("Matrix sigma is not Hermitian.")
    
    # Check for non-negative entries and trace one
    if np.any(np.diag(rho) < 0) or np.any(np.diag(sigma) < 0):
        raise ValueError("Density matrices must have non-negative entries.")
    if not np.isclose(np.trace(rho), 1) or not np.isclose(np.trace(sigma), 1):
        raise ValueError("Density matrices must have a trace of one.")

    # Compute the square root of rho
    sqrt_rho = sqrtm(rho)
    
    # Compute the product sqrt(rho) * sigma * sqrt(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    
    # Compute the square root of the product
    sqrt_product = sqrtm(product)
    
    # Compute the trace of the square root of the product
    trace_value = np.trace(sqrt_product)
    
    # Compute the fidelity
    fid = (trace_value.real) ** 2
    
    return fid



# Background: In quantum information theory, the fidelity between two quantum states is a measure of how similar they are. 
# For a given quantum protocol, such as the one described by the ghz_protocol function, we can calculate the fidelity of the 
# output state with respect to a target state, which in this case is a two-qubit maximally entangled state. The maximally 
# entangled state for two qubits is often the Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2. The fidelity calculation involves applying 
# the quantum channels to the input state, using the protocol to post-select the desired state, and then comparing the resulting 
# state to the target maximally entangled state using the fidelity formula. This process helps in assessing the effectiveness 
# of the protocol in generating entangled states.

def ghz_protocol_fidelity(input_state, channel1, channel2=None):
    '''Returns the achievable fidelity of the protocol
    Inputs:
        input_state: density matrix of the input 2n qubit state, ( 2**(2n), 2**(2n) ) array of floats
        channel1: kraus operators of the first channel, list of (2,2) array of floats
        channel2: kraus operators of the second channel, list of (2,2) array of floats
    Output:
        fid: achievable fidelity of protocol, float
    '''



    # Determine the number of qubits n
    total_dim = input_state.shape[0]
    n = int(np.log2(total_dim) / 2)
    
    # If channel2 is None, set it to channel1
    if channel2 is None:
        channel2 = channel1
    
    # Apply the channels to the input state
    output_state = channel_output(input_state, channel1, channel2)
    
    # Apply the GHZ protocol to the output state
    post_selected_state = ghz_protocol(output_state)
    
    # Define the target maximally entangled state |Φ+⟩ = (|00⟩ + |11⟩) / √2
    phi_plus = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
    
    # Calculate the fidelity between the post-selected state and the target state
    fid = fidelity(post_selected_state, phi_plus)
    
    return fid

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('65.6', 5)
target = targets[0]

ghz = np.zeros(16); ghz[0]=1/np.sqrt(2); ghz[-1]=1/np.sqrt(2); ghz = np.outer(ghz,ghz)
dephasing = [np.array([[np.sqrt(0.8),0],[0,np.sqrt(0.8)]]),
            np.array([[np.sqrt(0.2),0],[0,-np.sqrt(0.2)]])]
assert np.allclose(ghz_protocol_fidelity(ghz,dephasing), target)
target = targets[1]

ghz = np.zeros(16); ghz[0]=1/np.sqrt(2); ghz[-1]=1/np.sqrt(2); ghz = np.outer(ghz,ghz)
bitflip = [np.array([[np.sqrt(0.8),0],[0,np.sqrt(0.8)]]),
            np.array([[0,np.sqrt(0.2)],[np.sqrt(0.2),0]])]
assert np.allclose(ghz_protocol_fidelity(ghz,bitflip), target)
target = targets[2]

ghz = np.zeros(16); ghz[0]=1/np.sqrt(2); ghz[-1]=1/np.sqrt(2); ghz = np.outer(ghz,ghz)
y_error = [np.array([[np.sqrt(0.9),0],[0,np.sqrt(0.9)]]),
            np.array([[0,-1j*np.sqrt(0.1)],[1j*np.sqrt(0.1),0]])]
assert np.allclose(ghz_protocol_fidelity(ghz,y_error), target)
target = targets[3]

ghz = np.zeros(16); ghz[0]=1/np.sqrt(2); ghz[-1]=1/np.sqrt(2); ghz = np.outer(ghz,ghz)
identity = [np.eye(2)]
assert np.allclose(ghz_protocol_fidelity(ghz,identity), target)
target = targets[4]

ghz = np.zeros(16); ghz[0]=1/np.sqrt(2); ghz[-1]=1/np.sqrt(2); ghz = np.outer(ghz,ghz)
comp_dephasing = [np.array([[np.sqrt(0.5),0],[0,np.sqrt(0.5)]]),
                  np.array([[np.sqrt(0.5),0],[0,-np.sqrt(0.5)]])]
assert np.allclose(ghz_protocol_fidelity(ghz,comp_dephasing), target)
