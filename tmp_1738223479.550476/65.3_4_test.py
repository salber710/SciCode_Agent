from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.linalg import sqrtm
import itertools


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one input is required")
    
    # Convert inputs to numpy arrays and store their shapes
    matrices = [np.asarray(arg) for arg in args]
    shapes = [matrix.shape for matrix in matrices]

    # Calculate the shape of the resulting tensor product
    result_shape = (np.prod([shape[0] for shape in shapes]), np.prod([shape[1] for shape in shapes]))

    # Initialize the result as a 1 element matrix containing 1.0
    M = np.array([[1.0]])

    # Compute the Kronecker product using a unique approach with np.add.outer
    for matrix in matrices:
        # Use add.outer to generate a block matrix for the Kronecker product
        M = np.add.outer(M, matrix).reshape(M.shape[0] * matrix.shape[0], M.shape[1] * matrix.shape[1])
    
    return M


def apply_channel(K, rho, sys=None, dim=None):

    
    def full_operator(K_i, active_index, dims):
        """Constructs the full operator by embedding Kraus operator into the full space."""
        identities = [np.eye(d, dtype=np.complex128) for d in dims]
        identities[active_index] = K_i
        return np.linalg.multi_dot([np.kron(np.eye(np.prod(dims[:i]), dtype=np.complex128), identities[i]) for i in range(len(dims))])
    
    if sys is None and dim is None:
        # Apply the channel to the entire system
        return np.sum([K_i @ rho @ K_i.conj().T for K_i in K], axis=0)

    if sys is None or dim is None:
        raise ValueError("Both sys and dim must be provided if not applying to the entire system")
    
    total_dim = int(np.sqrt(rho.size))
    if total_dim != np.prod(dim):
        raise ValueError("Dimensions do not match the size of rho")

    new_rho = np.zeros_like(rho, dtype=np.complex128)
    
    for K_i in K:
        for idx in sys:
            op = full_operator(K_i, idx, dim)
            new_rho += op @ rho @ op.conj().T
    
    return new_rho



def channel_output(input_state, channel1, channel2=None):
    '''Returns the channel output
    Inputs:
        input_state: density matrix of the input 2n qubit state, ( 2**(2n), 2**(2n) ) array of floats
        channel1: kraus operators of the first channel, list of (2,2) array of floats
        channel2: kraus operators of the second channel, list of (2,2) array of floats
    Output:
        output: the channel output, ( 2**(2n), 2**(2n) ) array of floats
    '''

    # Assign channel2 to channel1 if channel2 is None using Python's 'is' keyword
    if channel2 is None:
        channel2 = channel1

    # Calculate the number of qubits n in each subsystem using logarithm and base 2

    size = input_state.shape[0]
    n = int(math.log2(size) / 2)

    # Apply channel1 to the first n qubits using a higher-order function
    def apply_to_first_half(state):
        return apply_channel(channel1, state, sys=[i for i in range(n)], dim=[2] * n)
    
    intermediate_state = apply_to_first_half(input_state)

    # Apply channel2 to the last n qubits using a higher-order function
    def apply_to_second_half(state):
        return apply_channel(channel2, state, sys=[n + i for i in range(n)], dim=[2] * n)
    
    output = apply_to_second_half(intermediate_state)

    return output


try:
    targets = process_hdf5_to_tuple('65.3', 3)
    target = targets[0]
    input_state = np.array([[1,0,0,1],
                           [0,0,0,0],
                           [0,0,0,0],
                           [1,0,0,1]])/2
    channel1 = [np.array([[0,1],[1,0]])]
    assert np.allclose(channel_output(input_state,channel1), target)

    target = targets[1]
    input_state = np.array([[0.8,0,0,np.sqrt(0.16)],
                           [0,0,0,0],
                           [0,0,0,0],
                           [np.sqrt(0.16),0,0,0.2]])
    channel1 = [np.array([[np.sqrt(0.5),0],[0,np.sqrt(0.5)]]),
                np.array([[np.sqrt(0.5),0],[0,-np.sqrt(0.5)]])]
    assert np.allclose(channel_output(input_state,channel1), target)

    target = targets[2]
    input_state = np.array([[1,0,0,1],
                           [0,0,0,0],
                           [0,0,0,0],
                           [1,0,0,1]])/2
    channel1 = [np.array([[0,1],[1,0]])]
    channel2 = [np.eye(2)]
    assert np.allclose(channel_output(input_state,channel1,channel2), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e