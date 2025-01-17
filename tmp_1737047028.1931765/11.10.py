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


# Background: In quantum mechanics, a multipartite quantum state can be represented as a density matrix, 
# which is a 2D array. When dealing with multipartite systems, each subsystem can have a different dimension. 
# Permuting the subsystems of a state involves rearranging the order of these subsystems according to a specified 
# permutation. This is useful in various quantum information tasks where the order of subsystems matters. 
# The permutation is applied to both the rows and columns of the density matrix, and the dimensions of the 
# subsystems are used to correctly identify the blocks of the matrix that correspond to each subsystem.

def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    perm: list of int containing the desired order
    dim: list of int containing the dimensions of all subsystems.
    Output:
    Y: 2d array of floats with equal dimensions, the density matrix of the permuted state
    '''
    # Calculate the number of subsystems
    num_subsystems = len(dim)
    
    # Calculate the total dimension of the system
    total_dim = np.prod(dim)
    
    # Reshape X into a tensor with shape (dim[0], dim[1], ..., dim[0], dim[1], ...)
    # The first num_subsystems dimensions are for the rows, the next num_subsystems are for the columns
    tensor_shape = dim + dim
    X_tensor = X.reshape(tensor_shape)
    
    # Create the permutation for the tensor
    # We need to permute both the row and column indices
    permuted_indices = perm + [p + num_subsystems for p in perm]
    
    # Apply the permutation to the tensor
    Y_tensor = np.transpose(X_tensor, permuted_indices)
    
    # Reshape the permuted tensor back into a matrix
    Y = Y_tensor.reshape((total_dim, total_dim))
    
    return Y


# Background: In quantum mechanics, the partial trace is an operation used to trace out (or discard) certain subsystems 
# of a multipartite quantum state, resulting in a reduced density matrix for the remaining subsystems. This is useful 
# when we are interested in the state of a subset of a larger quantum system. The partial trace is performed by summing 
# over the degrees of freedom of the subsystems to be traced out. The syspermute function can be used to rearrange the 
# subsystems so that the ones to be traced out are contiguous, simplifying the tracing process.

def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''



    # Calculate the number of subsystems
    num_subsystems = len(dim)
    
    # Calculate the total dimension of the system
    total_dim = np.prod(dim)
    
    # Determine the subsystems to keep
    keep = [i for i in range(num_subsystems) if i not in sys]
    
    # Permute the system so that the subsystems to be traced out are at the end
    perm = keep + sys
    X_permuted = syspermute(X, perm, dim)
    
    # Calculate the dimensions of the subsystems to keep and trace out
    dim_keep = [dim[i] for i in keep]
    dim_trace = [dim[i] for i in sys]
    
    # Reshape the permuted matrix into a tensor
    tensor_shape = dim_keep + dim_trace + dim_keep + dim_trace
    X_tensor = X_permuted.reshape(tensor_shape)
    
    # Perform the partial trace by summing over the trace dimensions
    Y_tensor = np.trace(X_tensor, axis1=len(dim_keep), axis2=len(dim_keep) + len(dim_trace))
    
    # Reshape the result back into a matrix
    Y = Y_tensor.reshape((np.prod(dim_keep), np.prod(dim_keep)))
    
    return Y



# Background: The von Neumann entropy is a measure of the quantum information content of a quantum state, analogous to the Shannon entropy in classical information theory. It is defined for a quantum state represented by a density matrix ρ as S(ρ) = -Tr(ρ log2 ρ), where Tr denotes the trace operation. The entropy quantifies the amount of uncertainty or mixedness of the quantum state. For pure states, the entropy is zero, while for mixed states, it is positive. The calculation involves finding the eigenvalues of the density matrix, as these represent the probabilities of the state being in each of its eigenstates. The entropy is then computed using these eigenvalues.

def entropy(rho):
    '''Inputs:
    rho: 2d array of floats with equal dimensions, the density matrix of the state
    Output:
    en: quantum (von Neumann) entropy of the state rho, float
    '''
    # Import necessary dependencies



    # Compute the eigenvalues of the density matrix
    eigenvalues = scipy.linalg.eigvalsh(rho)

    # Filter out zero eigenvalues to avoid log(0)
    nonzero_eigenvalues = eigenvalues[eigenvalues > 0]

    # Calculate the von Neumann entropy using the formula S(ρ) = -Tr(ρ log2 ρ)
    en = -np.sum(nonzero_eigenvalues * np.log2(nonzero_eigenvalues))

    return en


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('11.10', 3)
target = targets[0]

rho = np.eye(4)/4
assert np.allclose(entropy(rho), target)
target = targets[1]

rho = np.ones((3,3))/3
assert np.allclose(entropy(rho), target)
target = targets[2]

rho = np.diag([0.8,0.2])
assert np.allclose(entropy(rho), target)
