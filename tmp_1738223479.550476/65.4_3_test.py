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

    # Use a lambda expression to default channel2 to channel1
    channel2 = (lambda c1, c2: c1 if c2 is None else c2)(channel1, channel2)

    # Calculate the number of qubits n in each subsystem using a list comprehension
    n = int((len(bin(input_state.shape[0])) - 2) / 2)

    # Using map and lambda to create systems for channel application
    first_system = list(map(lambda i: i, range(n)))
    second_system = list(map(lambda i: i + n, range(n)))

    # Nested function to encapsulate the application of channels
    def apply_channels(state, channels, systems):
        for channel, system in zip(channels, systems):
            state = apply_channel(channel, state, sys=system, dim=[2] * n)
        return state

    # Apply the channels sequentially
    output = apply_channels(input_state, [channel1, channel2], [first_system, second_system])

    return output




def ghz_protocol(state):
    '''Returns the output state of the protocol
    Input:
    state: 2n qubit input state, 2^2n by 2^2n array of floats, where n is determined from the size of the input state
    Output:
    post_selected: the output state
    '''

    # Determine number of qubits per party
    total_qubits = int(np.log2(state.shape[0]))
    n = total_qubits // 2

    # Initialize an empty 2x2 matrix for the post-selected state
    post_selected = np.zeros((2, 2), dtype=np.float64)

    # Define the indices corresponding to even parity for both halves
    even_parity_indices = [(0, 0), (0, -1), (-1, 0), (-1, -1)]

    # Iterate over all possible combinations of even parity indices
    for idx1, idx2 in even_parity_indices:
        # Calculate the index in the full matrix
        index = (idx1 * (2**n) + idx2)
        post_selected[idx1//(2**(n-1)), idx2//(2**(n-1))] += state[index, index]

    # Normalize the post-selected state if it is non-zero
    normalization = np.trace(post_selected)
    if normalization > 0:
        post_selected /= normalization

    return post_selected


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e