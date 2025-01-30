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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('65.4', 3)
target = targets[0]

state = np.zeros(16); state[0]=np.sqrt(0.8); state[-1]=np.sqrt(0.2); state = np.outer(state,state)
assert np.allclose(ghz_protocol(state), target)
target = targets[1]

state = np.diag([0.3,0.05,0,0.05,0.05,0,0,0.05]*2)
assert np.allclose(ghz_protocol(state), target)
target = targets[2]

state = np.ones((16,16))/16
assert np.allclose(ghz_protocol(state), target)
