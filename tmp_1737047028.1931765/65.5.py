import numpy as np
from scipy.linalg import sqrtm
import itertools

# Background: The tensor product, also known as the Kronecker product, is an operation on two matrices (or vectors) that results in a block matrix. 
# If A is an m x n matrix and B is a p x q matrix, their Kronecker product A ⊗ B is an mp x nq matrix. 
# The Kronecker product is a generalization of the outer product from vectors to matrices. 
# It is used in various fields such as quantum computing, signal processing, and the study of multi-linear algebra. 
# In this function, we aim to compute the tensor product of an arbitrary number of matrices or vectors.


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
    
    # Iterate over the remaining matrices/vectors and compute the Kronecker product
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M


# Background: In quantum mechanics, a quantum channel is a mathematical model for the physical process of transmitting quantum states. 
# It is represented by a set of Kraus operators {K_i}, which are used to describe the evolution of a quantum state ρ. 
# The action of a quantum channel on a state ρ is given by ρ' = Σ_i K_i ρ K_i†, where K_i† is the conjugate transpose of K_i. 
# When applying a quantum channel to specific subsystems of a composite quantum system, we need to consider the tensor product structure of the state. 
# The tensor function can be used to construct the appropriate operators for the subsystems. 
# If the channel is applied to the entire system, the dimensions of the subsystems are not needed. 
# However, if the channel is applied to specific subsystems, the dimensions of these subsystems must be specified to correctly apply the channel.



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
        new_rho = np.zeros_like(rho)
        for k in K:
            new_rho += k @ rho @ k.conj().T
        return new_rho
    else:
        # Apply the channel to specified subsystems
        total_dim = np.prod(dim)
        if rho.shape != (total_dim, total_dim):
            raise ValueError("The dimensions of rho do not match the product of the subsystem dimensions.")
        
        # Create identity operators for subsystems not in sys
        identity_ops = [np.eye(d) for d in dim]
        
        # Initialize the new density matrix
        new_rho = np.zeros_like(rho)
        
        # Iterate over all Kraus operators
        for k in K:
            # Construct the full operator for the entire system
            full_operator = [identity_ops[i] if i not in sys else k for i in range(len(dim))]
            full_operator = np.kron.reduce(full_operator)
            
            # Apply the operator to the state
            new_rho += full_operator @ rho @ full_operator.conj().T
        
        return new_rho


# Background: In quantum information theory, a quantum channel is a model for the transmission of quantum states. 
# When dealing with multi-qubit systems, it is common to apply different quantum channels to different parts of the system. 
# In this problem, we have a 2n-qubit input state, and we want to apply one quantum channel (channel1) to the first n qubits 
# and another quantum channel (channel2) to the last n qubits. If channel2 is not provided, it defaults to channel1. 
# The apply_channel function is used to apply the Kraus operators of a quantum channel to a specified subsystem of a quantum state. 
# The task is to implement a function that returns the output state after applying these channels to the respective qubits.

def channel_output(input_state, channel1, channel2=None):
    '''Returns the channel output
    Inputs:
        input_state: density matrix of the input 2n qubit state, ( 2**(2n), 2**(2n) ) array of floats
        channel1: kraus operators of the first channel, list of (2,2) array of floats
        channel2: kraus operators of the second channel, list of (2,2) array of floats
    Output:
        output: the channel output, ( 2**(2n), 2**(2n) ) array of floats
    '''

    
    # If channel2 is not provided, set it to be the same as channel1
    if channel2 is None:
        channel2 = channel1
    
    # Determine the number of qubits n
    n = input_state.shape[0] // 2
    
    # Apply channel1 to the first n qubits
    output_state = apply_channel(channel1, input_state, sys=list(range(n)), dim=[2]*2*n)
    
    # Apply channel2 to the last n qubits
    output_state = apply_channel(channel2, output_state, sys=list(range(n, 2*n)), dim=[2]*2*n)
    
    return output_state


# Background: In quantum mechanics, a parity measurement is a type of measurement that checks whether the number of qubits in the state |1⟩ is even or odd. 
# For a set of n qubits, an even parity means that the number of qubits in the state |1⟩ is even (including zero), and an odd parity means it is odd. 
# In this protocol, we perform parity measurements on two sets of n qubits. If both sets have even parity, the state is kept and transformed into a two-qubit state. 
# Specifically, the state |00...0⟩ (n 0's) is transformed into |0⟩, and the state |11...1⟩ (n 1's) is transformed into |1⟩. 
# If either set has odd parity, the state is discarded. This process is a form of post-selection, where only certain measurement outcomes are kept.



def ghz_protocol(state):
    '''Returns the output state of the protocol
    Input:
    state: 2n qubit input state, 2^2n by 2^2n array of floats, where n is determined from the size of the input state
    Output:
    post_selected: the output state
    '''
    # Determine the number of qubits n
    dim = state.shape[0]
    n = dim.bit_length() // 2

    # Generate all possible binary strings of length n
    binary_strings = [''.join(seq) for seq in itertools.product('01', repeat=n)]

    # Define the indices for even parity states
    even_parity_indices = [i for i, b in enumerate(binary_strings) if b.count('1') % 2 == 0]

    # Initialize the post-selected state
    post_selected = np.zeros((2, 2), dtype=complex)

    # Iterate over all combinations of even parity indices for both sets of n qubits
    for i in even_parity_indices:
        for j in even_parity_indices:
            # Calculate the index in the original 2^2n space
            index = i * (2**n) + j
            # Add the contribution to the post-selected state
            post_selected[0, 0] += state[index, index] if i == 0 and j == 0 else 0
            post_selected[1, 1] += state[index, index] if i == (2**n - 1) and j == (2**n - 1) else 0

    # Normalize the post-selected state
    trace = np.trace(post_selected)
    if trace > 0:
        post_selected /= trace

    return post_selected



# Background: In quantum information theory, fidelity is a measure of the "closeness" or similarity between two quantum states. 
# It is particularly useful for comparing a theoretical quantum state with an experimentally prepared one. 
# The fidelity F between two density matrices ρ and σ is defined as F(ρ, σ) = (Tr(sqrt(sqrt(ρ) σ sqrt(ρ))))^2. 
# This formula involves taking the square root of the density matrices and computing the trace of their product. 
# Fidelity ranges from 0 to 1, where 1 indicates that the states are identical, and 0 indicates that they are orthogonal.

def fidelity(rho, sigma):
    '''Returns the fidelity between two states.
    Inputs:
    rho, sigma: density matrices of the two states, 2d array of floats
    Output:
    fid: fidelity, float
    '''
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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('65.5', 3)
target = targets[0]

rho = np.array([[1,0,0,1],
                [0,0,0,0],
                [0,0,0,0],
                [1,0,0,1]])/2
sigma = np.eye(4)/4
assert np.allclose(fidelity(rho,sigma), target)
target = targets[1]

rho = np.array([[1/2,1/2],[1/2,1/2]])
sigma = np.array([[1,0],[0,0]])
assert np.allclose(fidelity(rho,sigma), target)
target = targets[2]

rho = np.array([[1/2,1/2],[1/2,1/2]])
sigma = np.array([[1/2,-1/2],[-1/2,1/2]])
assert np.allclose(fidelity(rho,sigma), target)
