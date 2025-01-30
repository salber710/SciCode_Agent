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
        # Define the Kraus operators for generalized amplitude damping
        E0 = np.sqrt(1 - gamma) * np.array([[1, 0], [0, np.sqrt(1 - N)]])
        E1 = np.sqrt(gamma) * np.array([[np.sqrt(1 - N), 0], [0, 1]])
        E2 = np.sqrt(1 - gamma) * np.array([[0, np.sqrt(N)], [0, 0]])
        E3 = np.sqrt(gamma) * np.array([[0, 0], [np.sqrt(N), 0]])
        return [E0, E1, E2, E3]

    def apply_channel(state, kraus_ops):
        # Apply the channel defined by the Kraus operators to the state
        new_state = np.zeros_like(state)
        for E in kraus_ops:
            new_state += E @ state @ E.conj().T
        return new_state

    def initial_state(rails):
        # Create a maximally entangled state for 'rails' qubits
        size = 2 ** rails
        state_vector = np.ones(size) / np.sqrt(size)
        return np.outer(state_vector, state_vector)

    # Initialize the state
    state = initial_state(rails)
    
    # Get Kraus operators for both channels
    kraus_1 = kraus_operators(gamma_1, N_1)
    kraus_2 = kraus_operators(gamma_2, N_2)

    # Sequentially apply the first and second set of Kraus operators to the state
    state = apply_channel(state, kraus_1)
    state = apply_channel(state, kraus_2)

    return state



def measurement(rails):
    '''Returns the measurement projector
    Input:
    rails: int, number of rails
    Output:
    global_proj: (2**(2*rails), 2**(2*rails)) dimensional array of floats
    '''
    # Dimension of the Hilbert space for one set of rails
    dim = 2**rails
    
    # Create a single-rail projector using a functional programming approach
    single_rail_projector = np.sum(np.array([np.outer(np.eye(1, dim, 1 << i), np.eye(1, dim, 1 << i).T) for i in range(rails)]), axis=0)
    
    # The global projector is the Kronecker product of the single-rail projector with itself
    global_proj = np.kron(single_rail_projector, single_rail_projector)
    
    return global_proj


def syspermute(X, perm, dim):


    # Calculate the total dimension of the system
    total_dim = np.prod(dim)
    
    # Ensure the input matrix X is square and matches the total dimension
    assert X.shape == (total_dim, total_dim), "Input matrix X must be square and match the total dimension."

    # Calculate the number of subsystems
    num_subsystems = len(dim)

    # Reshape X into a tensor with dimensions for rows and columns of each subsystem
    reshaped_dims = [dim[i] for i in range(num_subsystems)] * 2
    X_reshaped = X.reshape(reshaped_dims)

    # Create new order for axes, considering rows and columns for each subsystem
    new_order = [perm[i] * 2 for i in range(num_subsystems)] + [(perm[i] * 2) + 1 for i in range(num_subsystems)]

    # Permute the axes of the reshaped matrix
    X_permuted = np.transpose(X_reshaped, new_order)

    # Flatten the permuted matrix back to 2D
    Y = X_permuted.reshape(total_dim, total_dim)
    
    return Y



def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''
    # Calculate the total dimension of the system
    total_dim = np.prod(dim)
    
    # Ensure the input matrix X is square and matches the total dimension
    assert X.shape == (total_dim, total_dim), "Input matrix X must be square and match the total dimension."

    # Calculate the number of subsystems
    num_subsystems = len(dim)

    # Determine the subsystems to keep
    keep = [i for i in range(num_subsystems) if i not in sys]

    # Permute the system so that the subsystems to be traced out are at the end
    perm = keep + sys
    X_permuted = syspermute(X, perm, dim)

    # Calculate the dimensions of the subsystems to keep and trace out
    dim_keep = [dim[i] for i in keep]
    dim_trace = [dim[i] for i in sys]

    # Reshape the permuted matrix to separate the subsystems
    reshaped_dims = dim_keep + dim_trace + dim_keep + dim_trace
    X_reshaped = X_permuted.reshape(reshaped_dims)

    # Perform the partial trace by summing over the traced out subsystems
    for i in range(len(dim_trace)):
        X_reshaped = np.trace(X_reshaped, axis1=len(dim_keep)+i, axis2=len(dim_keep)+len(dim_trace)+i)

    # Reshape back to the correct dimensions
    final_dim = np.prod(dim_keep)
    return X_reshaped.reshape(final_dim, final_dim)



def entropy(rho):
    # Calculate the eigenvalues of the density matrix
    eigenvalues = np.linalg.eigvalsh(rho)
    
    # Compute the entropy using numpy's vectorized operations and ensuring numerical stability with np.maximum
    entropy_values = -eigenvalues * np.log2(np.maximum(eigenvalues, np.finfo(float).tiny))
    en = np.sum(entropy_values)
    
    return en




def coherent_inf_state(rho_AB, dimA, dimB):
    # Function to compute the entropy using the base-e logarithm and numerical stability
    def entropy(matrix):
        eigvals = np.linalg.eigvalsh(matrix)
        eigvals = eigvals[eigvals > 1e-10]  # Filter out very small eigenvalues
        return -np.sum(eigvals * np.log(eigvals))

    # Compute the reduced state rho_B by averaging over blocks corresponding to system A
    rho_B = np.sum(rho_AB.reshape(dimA, dimB, dimA, dimB), axis=(0, 2)) / dimA

    # Calculate the entropies
    S_AB = entropy(rho_AB)
    S_B = entropy(rho_B)

    # Calculate the coherent information
    co_inf = S_B - S_AB

    return co_inf


try:
    targets = process_hdf5_to_tuple('11.11', 3)
    target = targets[0]
    rho_AB = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    assert np.allclose(coherent_inf_state(rho_AB, 2, 2), target)

    target = targets[1]
    rho_AB = np.eye(6)/6
    assert np.allclose(coherent_inf_state(rho_AB, 2, 3), target)

    target = targets[2]
    rho_AB = np.diag([1,0,0,1])
    assert np.allclose(coherent_inf_state(rho_AB, 2, 2), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e