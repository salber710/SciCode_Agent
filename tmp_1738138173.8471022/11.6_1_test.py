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




def output_state(rails, gamma_1, N_1, gamma_2, N_2):
    def kraus_operators(gamma, N):
        E0 = np.sqrt(1 - gamma) * np.array([[1, 0], [0, np.sqrt(1 - N)]])
        E1 = np.sqrt(1 - gamma) * np.array([[0, np.sqrt(N)], [0, 0]])
        E2 = np.sqrt(gamma) * np.array([[np.sqrt(1 - N), 0], [0, 1]])
        E3 = np.sqrt(gamma) * np.array([[0, 0], [np.sqrt(N), 0]])
        return [E0, E1, E2, E3]

    def apply_channel(state, kraus_ops):
        d = state.shape[0]
        new_state = np.zeros((d, d), dtype=complex)
        for E in kraus_ops:
            new_state += E @ state @ E.conj().T
        return new_state

    def initial_state(rails):
        size = 2 ** rails
        state_vector = np.zeros(size)
        state_vector[0] = 1
        return np.outer(state_vector, state_vector)

    state = initial_state(rails)
    kraus_1 = kraus_operators(gamma_1, N_1)
    kraus_2 = kraus_operators(gamma_2, N_2)

    for i in range(rails):
        indices = [2 * i, 2 * i + 1]
        for j in indices:
            state = np.kron(np.eye(2 ** j), np.kron(apply_channel(state[2 ** j: 2 ** (j + 1), 2 ** j: 2 ** (j + 1)], kraus_1), np.eye(2 ** (2 * rails - j - 1))))
        for j in indices:
            state = np.kron(np.eye(2 ** j), np.kron(apply_channel(state[2 ** j: 2 ** (j + 1), 2 ** j: 2 ** (j + 1)], kraus_2), np.eye(2 ** (2 * rails - j - 1))))

    return state


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