from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg

def ket(dim, args):


    def create_basis_vector(d, i):
        # Create a d-dimensional zero vector and set the i-th position to 1
        vec = np.zeros(d)
        vec[i] = 1
        return vec

    if isinstance(args, list):
        # Compute the tensor product of basis vectors for each dimension and index in args
        result = create_basis_vector(dim[0], args[0])
        for d, i in zip(dim[1:], args[1:]):
            result = np.kron(result, create_basis_vector(d, i))
    else:
        # Single dimension and index
        result = create_basis_vector(dim, args)

    return result



def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    # Dimension of each subsystem
    dim = 2 ** rails
    
    # Create a maximally entangled state using a Hadamard-like approach
    entangled_state = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            entangled_state[i, j] = 1 if (i ^ j) == 0 else 0
    
    # Flatten the matrix to create the state vector
    entangled_state = entangled_state.flatten()
    
    # Normalize the state vector
    entangled_state /= np.sqrt(dim)
    
    # Create the density matrix by taking the outer product of the state vector with itself
    state = np.outer(entangled_state, entangled_state)
    
    return state


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''


    # Compute the tensor product using a loop, but process in pairs from end to start
    if not args:
        return np.array([[]])  # Return an empty array if no arguments are provided

    # Start with the last matrix/vector
    M = args[-1]

    # Process in pairs
    for i in range(len(args) - 2, -1, -2):
        if i - 1 >= 0:
            M = np.kron(np.kron(args[i-1], args[i]), M)
        else:
            M = np.kron(args[i], M)

    return M


def apply_channel(K, rho, sys=None, dim=None):


    def tensor_product(*matrices):
        """Compute the tensor product of multiple matrices."""

        return reduce(np.kron, matrices)

    if sys is None or dim is None:
        # Apply channel to the entire system
        return np.sum([k @ rho @ k.conj().T for k in K], axis=0)

    # Initialize the resulting state
    result_rho = np.zeros_like(rho)
    identity_matrices = [np.eye(d) for d in dim]

    # Apply channel to specified subsystems
    for k in K:
        kraus_ops = [identity_matrices[i] if i not in sys else k for i in range(len(dim))]
        full_kraus_op = tensor_product(*kraus_ops)
        result_rho += full_kraus_op @ rho @ full_kraus_op.conj().T

    return result_rho


def generalized_amplitude_damping_channel(gamma, N):


    # Define the square roots of probabilities for the Kraus operators
    p0 = np.sqrt(gamma * (1 - N))
    p1 = np.sqrt(gamma * N)
    q0 = np.sqrt(1 - gamma)
    q1 = np.sqrt(1 - gamma * N)

    # Define the Kraus operators using a different matrix arrangement and data type
    A1 = np.array([[q0, 0], [0, q1]], dtype=np.float64)
    A2 = np.array([[0, 0], [0, p1]], dtype=np.float64)
    A3 = np.array([[q1, 0], [0, q0]], dtype=np.float64)
    A4 = np.array([[0, p0], [0, 0]], dtype=np.float64)

    # Return the list of Kraus operators
    kraus = [A1, A2, A3, A4]

    return kraus



# Background: In quantum information theory, a generalized amplitude damping channel is a model for noise that describes the interaction of a quantum system with a thermal environment. It is characterized by two parameters: the damping parameter γ, which represents the probability of energy loss, and the thermal parameter N, which represents the average number of thermal excitations in the environment. The Kraus operators for this channel describe how the quantum state is transformed when it passes through the channel. In this task, we need to apply these channels to a multi-rail encoded quantum state, which is a type of entangled state, and compute the resulting state after the noise has been applied.


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
    
    def generalized_amplitude_damping_channel(gamma, N):
        p0 = np.sqrt(gamma * (1 - N))
        p1 = np.sqrt(gamma * N)
        q0 = np.sqrt(1 - gamma)
        q1 = np.sqrt(1 - gamma * N)

        A1 = np.array([[q0, 0], [0, q1]], dtype=np.float64)
        A2 = np.array([[0, 0], [0, p1]], dtype=np.float64)
        A3 = np.array([[q1, 0], [0, q0]], dtype=np.float64)
        A4 = np.array([[0, p0], [0, 0]], dtype=np.float64)

        return [A1, A2, A3, A4]

    def apply_channel(K, rho, sys=None, dim=None):
        if sys is None or dim is None:
            return np.sum([k @ rho @ k.conj().T for k in K], axis=0)

        result_rho = np.zeros_like(rho)
        identity_matrices = [np.eye(d) for d in dim]

        for k in K:
            kraus_ops = [identity_matrices[i] if i not in sys else k for i in range(len(dim))]
            full_kraus_op = np.kron(*kraus_ops)
            result_rho += full_kraus_op @ rho @ full_kraus_op.conj().T

        return result_rho

    def multi_rail_encoding_state(rails):
        dim = 2 ** rails
        entangled_state = np.zeros((dim, dim), dtype=np.float64)
        for i in range(dim):
            for j in range(dim):
                entangled_state[i, j] = 1 if (i ^ j) == 0 else 0
        entangled_state = entangled_state.flatten()
        entangled_state /= np.sqrt(dim)
        state = np.outer(entangled_state, entangled_state)
        return state

    # Generate the initial multi-rail encoded state
    initial_state = multi_rail_encoding_state(rails)

    # Get the Kraus operators for both channels
    K1 = generalized_amplitude_damping_channel(gamma_1, N_1)
    K2 = generalized_amplitude_damping_channel(gamma_2, N_2)

    # Apply the channels to each rail
    dim = [2] * (2 * rails)
    sys1 = list(range(rails))
    sys2 = list(range(rails, 2 * rails))

    # Apply the first set of channels to the first half of the rails
    state_after_first_channel = apply_channel(K1, initial_state, sys=sys1, dim=dim)

    # Apply the second set of channels to the second half of the rails
    final_state = apply_channel(K2, state_after_first_channel, sys=sys2, dim=dim)

    return final_state


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