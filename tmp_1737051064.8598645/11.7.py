import numpy as np
import itertools
import scipy.linalg
def ket(dim, args):
    """Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    """
    if isinstance(dim, int):
        out = np.zeros(dim)
        out[args] = 1.0
    else:
        vectors = []
        for (d, j) in zip(dim, args):
            vec = np.zeros(d)
            vec[j] = 1.0
            vectors.append(vec)
        out = vectors[0]
        for vec in vectors[1:]:
            out = np.kron(out, vec)
    return out
def multi_rail_encoding_state(rails):
    """Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    """
    dim = 2 ** rails
    entangled_state = np.zeros((dim * dim,), dtype=np.float64)
    for i in range(dim):
        ket_i = ket(dim, i)
        ket_j = ket(dim, i)
        tensor_product = np.kron(ket_i, ket_j)
        entangled_state += tensor_product
    entangled_state /= np.sqrt(dim)
    state = np.outer(entangled_state, entangled_state)
    return state
def tensor(*args):
    """Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    """
    M = args[0]
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    return M
def apply_channel(K, rho, sys=None, dim=None):
    """Applies the channel with Kraus operators in K to the state rho on
    systems specified by the list sys. The dimensions of the subsystems of
    rho are given by dim.
    Inputs:
    K: list of 2d array of floats, list of Kraus operators
    rho: 2d array of floats, input density matrix
    sys: list of int or None, list of subsystems to apply the channel, None means full system
    dim: list of int or None, list of dimensions of each subsystem, None means full system
    Output:
    matrix: output density matrix of floats
    """
    if sys is None or dim is None:
        new_rho = np.zeros_like(rho)
        for k in K:
            new_rho += k @ rho @ k.conj().T
        return new_rho
    else:
        total_dim = np.prod(dim)
        rho = rho.reshape([total_dim, total_dim])
        id_matrices = [np.eye(d) for d in dim]
        new_rho = np.zeros_like(rho)
        for kraus_ops in itertools.product(K, repeat=len(sys)):
            op = np.eye(total_dim)
            for (idx, s) in enumerate(sys):
                op_s = np.eye(dim[s])
                op_s = kraus_ops[idx] @ op_s @ kraus_ops[idx].conj().T
                op = np.kron(op, op_s)
            new_rho += op @ rho @ op.conj().T
        return new_rho

# Background: The generalized amplitude damping channel is a quantum channel that models the interaction of a qubit with a thermal bath at non-zero temperature. It is characterized by two parameters: the damping parameter gamma, which represents the probability of energy dissipation, and the thermal parameter N, which represents the average number of excitations in the environment. The channel is described by four Kraus operators, which are 2x2 matrices that satisfy the completeness relation. These operators are used to describe the evolution of the quantum state under the influence of the channel.


def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''
    # Calculate the Kraus operators
    A1 = np.sqrt(gamma) * np.array([[1, 0], [0, np.sqrt(1 - N)]])
    A2 = np.sqrt(gamma) * np.array([[0, np.sqrt(N)], [0, 0]])
    A3 = np.sqrt(1 - gamma) * np.array([[np.sqrt(1 - N), 0], [0, 1]])
    A4 = np.sqrt(1 - gamma) * np.array([[0, 0], [np.sqrt(N), 0]])
    
    kraus = [A1, A2, A3, A4]
    return kraus


# Background: In quantum information theory, a generalized amplitude damping channel models the interaction of a qubit with a thermal environment. 
# It is characterized by two parameters: gamma (the probability of energy dissipation) and N (the average number of excitations in the environment).
# The channel is described by four Kraus operators that transform the quantum state. When a quantum state is sent through multiple such channels,
# the overall effect is the application of these operators to the state. In this problem, we are dealing with an m-rail encoded state, which is a 
# bipartite maximally entangled state. We need to apply m generalized amplitude damping channels to each part of this entangled state, 
# with different parameters for each receiver.

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



    # Function to generate the Kraus operators for the generalized amplitude damping channel
    def generalized_amplitude_damping_channel(gamma, N):
        A1 = np.sqrt(gamma) * np.array([[1, 0], [0, np.sqrt(1 - N)]])
        A2 = np.sqrt(gamma) * np.array([[0, np.sqrt(N)], [0, 0]])
        A3 = np.sqrt(1 - gamma) * np.array([[np.sqrt(1 - N), 0], [0, 1]])
        A4 = np.sqrt(1 - gamma) * np.array([[0, 0], [np.sqrt(N), 0]])
        return [A1, A2, A3, A4]

    # Function to apply the channel with Kraus operators to the state
    def apply_channel(K, rho, sys=None, dim=None):
        if sys is None or dim is None:
            new_rho = np.zeros_like(rho)
            for k in K:
                new_rho += k @ rho @ k.conj().T
            return new_rho
        else:
            total_dim = np.prod(dim)
            rho = rho.reshape([total_dim, total_dim])
            id_matrices = [np.eye(d) for d in dim]
            new_rho = np.zeros_like(rho)
            for kraus_ops in itertools.product(K, repeat=len(sys)):
                op = np.eye(total_dim)
                for (idx, s) in enumerate(sys):
                    op_s = np.eye(dim[s])
                    op_s = kraus_ops[idx] @ op_s @ kraus_ops[idx].conj().T
                    op = np.kron(op, op_s)
                new_rho += op @ rho @ op.conj().T
            return new_rho

    # Generate the initial m-rail encoded state
    def multi_rail_encoding_state(rails):
        dim = 2 ** rails
        entangled_state = np.zeros((dim * dim,), dtype=np.float64)
        for i in range(dim):
            ket_i = np.zeros(dim)
            ket_i[i] = 1.0
            tensor_product = np.kron(ket_i, ket_i)
            entangled_state += tensor_product
        entangled_state /= np.sqrt(dim)
        state = np.outer(entangled_state, entangled_state)
        return state

    # Generate the initial state
    initial_state = multi_rail_encoding_state(rails)

    # Generate Kraus operators for both channels
    K1 = generalized_amplitude_damping_channel(gamma_1, N_1)
    K2 = generalized_amplitude_damping_channel(gamma_2, N_2)

    # Apply the channels to the state
    dim = [2] * (2 * rails)
    state_after_first_channel = apply_channel(K1, initial_state, sys=list(range(rails)), dim=dim)
    final_state = apply_channel(K2, state_after_first_channel, sys=list(range(rails, 2 * rails)), dim=dim)

    return final_state



# Background: In quantum mechanics, a projector is an operator that projects a state onto a subspace. 
# For a system of qubits, the one-particle sector refers to the subspace where exactly one qubit is in the state |1⟩ 
# and the rest are in the state |0⟩. For m qubits, this corresponds to states like |100...0⟩, |010...0⟩, ..., |000...1⟩.
# The task is to construct a global projector for a bipartite system where each part has m qubits, and each part is 
# measured to be in the one-particle sector. This involves creating a projector for each part and then taking the 
# tensor product of these projectors to form the global projector.



def measurement(rails):
    '''Returns the measurement projector
    Input:
    rails: int, number of rails
    Output:
    global_proj: (2**(2*rails), 2**(2*rails)) dimensional array of floats
    '''
    # Dimension of the space for each set of rails
    dim = 2 ** rails
    
    # Create the projector for the one-particle sector for one set of rails
    one_particle_proj = np.zeros((dim, dim), dtype=np.float64)
    
    # Iterate over all possible binary strings of length 'rails'
    for bits in itertools.product([0, 1], repeat=rails):
        if sum(bits) == 1:  # Check if there is exactly one '1' in the string
            # Convert the binary string to an index
            index = sum(b * (2 ** i) for i, b in enumerate(reversed(bits)))
            # Set the corresponding element in the projector
            one_particle_proj[index, index] = 1.0
    
    # The global projector is the tensor product of the projectors for each set of rails
    global_proj = np.kron(one_particle_proj, one_particle_proj)
    
    return global_proj


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('11.7', 3)
target = targets[0]

assert np.allclose(measurement(1), target)
target = targets[1]

assert np.allclose(measurement(2), target)
target = targets[2]

assert np.allclose(measurement(3), target)
