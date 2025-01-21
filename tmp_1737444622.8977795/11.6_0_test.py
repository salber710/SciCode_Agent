import numpy as np
import itertools
import scipy.linalg

# Background: In quantum mechanics, a ket, denoted as |j⟩, is a vector in a complex vector space that represents the state of a quantum system. 
# When dealing with quantum systems of multiple particles or subsystems, the state is often represented as a tensor product of the states of 
# individual subsystems. In such scenarios, a standard basis vector in a d-dimensional space is a vector that has a 1 in the j-th position 
# and 0 elsewhere. If j is a list, the goal is to construct a larger vector space that is the tensor product of individual spaces specified 
# by dimensions in d. The tensor product combines the spaces into one larger space, and the corresponding ket represents a state in this 
# composite space.

def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''


    # Helper function to create a standard basis vector in a given dimension
    def standard_basis_vector(d, j):
        v = np.zeros(d)
        v[j] = 1
        return v
    
    if isinstance(dim, int) and isinstance(args, int):
        # Single dimension and single index
        return standard_basis_vector(dim, args)
    
    elif isinstance(dim, list) and isinstance(args, list):
        # Multiple dimensions and multiple indices
        if len(dim) != len(args):
            raise ValueError("Length of dimensions and indices must match")
        
        # Calculate the tensor product of basis vectors
        vectors = [standard_basis_vector(d, j) for d, j in zip(dim, args)]
        result = vectors[0]
        for vector in vectors[1:]:
            result = np.kron(result, vector)
        return result
    
    else:
        raise TypeError("Both dim and args should be either int or list")
    
    return out


# Background: In quantum computing, a bipartite maximally entangled state is a special quantum state involving two subsystems
# that are maximally correlated. A common example is the Bell state. In the context of $m$-rail encoding, each party's system
# is represented using multiple "rails" or qubits. The multi-rail encoding involves distributing the information across multiple
# qubits per party to potentially increase robustness against certain types of errors. The density matrix of such a state describes
# the statistical mixtures of quantum states that the system can be in. For a bipartite maximally entangled state with $m$-rail 
# encoding, the state is constructed in a larger Hilbert space, where each subsystem is represented by $m$ qubits.

def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''


    
    # Dimension of the system for each party
    dimension = 2 ** rails
    
    # Create the maximally entangled state |Φ⟩ = 1/√d * Σ |i⟩|i⟩
    entangled_state = np.zeros((dimension, dimension), dtype=np.float64)
    
    for i in range(dimension):
        entangled_state[i, i] = 1
    
    # Normalize the state
    entangled_state = entangled_state / np.sqrt(dimension)
    
    # Compute the density matrix ρ = |Φ⟩⟨Φ|
    density_matrix = np.outer(entangled_state.flatten(), entangled_state.flatten().conj())
    
    return density_matrix


# Background: In linear algebra and quantum mechanics, the tensor product (also known as the Kronecker product when applied to matrices)
# is an operation on two matrices (or vectors) of arbitrary size resulting in a block matrix. If A is an m×n matrix and B is a p×q
# matrix, then the Kronecker product A⊗B is an mp×nq matrix. In quantum mechanics, the tensor product is used to combine quantum states
# of different systems into a joint state, representing the combined system. This operation is essential when dealing with multi-particle
# systems or composite quantum systems. In this context, the tensor product allows for the representation of states in a higher-dimensional
# Hilbert space that encompasses the states of all subsystems.

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''



    # Start with the first element as the result
    M = args[0]

    # Iteratively compute the Kronecker product with the rest of the matrices/vectors
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M


# Background: In quantum mechanics, a quantum channel is a mathematical model used to describe the evolution of quantum states,
# particularly in open quantum systems where the system interacts with an environment. The evolution of a quantum state under a channel
# is often represented using Kraus operators, which are a set of matrices {K_i} that satisfy a completeness relation. The action of a
# quantum channel on a density matrix ρ is given by the transformation ρ' = Σ_i K_i ρ K_i†, where K_i† is the conjugate transpose of K_i.
# When dealing with composite systems, it is often necessary to apply the channel to specific subsystems, which requires considering
# the tensor product structure and appropriately embedding the Kraus operators into the larger Hilbert space.




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
    
    # If sys and dim are None, apply the channel to the entire system
    if sys is None and dim is None:
        # Apply the channel to the full density matrix
        new_rho = np.zeros_like(rho, dtype=np.float64)
        for kraus_operator in K:
            new_rho += kraus_operator @ rho @ kraus_operator.conj().T
        return new_rho
    
    # Otherwise, apply the channel to selected subsystems
    elif sys is not None and dim is not None:
        total_dim = np.prod(dim)  # Total dimension of the composite system
        new_rho = np.zeros_like(rho, dtype=np.float64)

        # Create the identity operators for unaffected subsystems
        identity_operators = [np.eye(d) for d in dim]

        # Iterate over all Kraus operators
        for kraus_operator in K:
            # Prepare the operator to apply to the entire system
            operator_list = [identity_operators[i] for i in range(len(dim))]
            
            # Replace identity with kraus_operator for the selected subsystems
            for i, subsystem in enumerate(sys):
                operator_list[subsystem] = kraus_operator

            # Calculate the tensor product of operators
            full_operator = operator_list[0]
            for op in operator_list[1:]:
                full_operator = np.kron(full_operator, op)

            # Apply the full operator to the density matrix
            new_rho += full_operator @ rho @ full_operator.conj().T
        
        return new_rho
    
    else:
        raise ValueError("Both sys and dim must be specified if one is provided.")


# Background: In quantum mechanics, the generalized amplitude damping channel is a model that describes the interaction of a quantum system
# with a thermal bath at finite temperature. It is characterized by two parameters: the damping parameter γ (gamma), which represents the
# probability of energy dissipation from the system to the environment, and the thermal parameter N, which is related to the mean photon
# number of the thermal bath. The generalized amplitude damping channel is represented by a set of Kraus operators {A1, A2, A3, A4} that
# satisfy the completeness relation Σ_i A_i†A_i = I. These operators model the probability of the system transitioning between different
# states due to interactions with the thermal environment.

def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''


    # Calculate probabilities
    p = gamma * (1 + N)
    q = gamma * N

    # Define the Kraus operators for the generalized amplitude damping channel
    A1 = np.array([[np.sqrt(1 - q), 0],
                   [0, np.sqrt(1 - p)]], dtype=np.float64)

    A2 = np.array([[0, np.sqrt(p)],
                   [0, 0]], dtype=np.float64)

    A3 = np.array([[np.sqrt(q), 0],
                   [0, 0]], dtype=np.float64)

    A4 = np.array([[0, 0],
                   [np.sqrt(1 - q), 0]], dtype=np.float64)

    kraus = [A1, A2, A3, A4]
    
    return kraus



# Background: In quantum information, a quantum state can undergo transformations when transmitted through a noisy channel.
# A common model for such a channel is the generalized amplitude damping channel, which is characterized by parameters gamma
# and N. When a quantum state is sent through a series of such channels, it experiences energy dissipation and thermal noise.
# In this task, we consider an m-rail encoded quantum state, which means that the state is encoded across multiple "rails" or
# subsystems. The goal is to model the transformation of this state as it passes through m generalized amplitude damping channels
# for each of the two receivers. The output state is obtained by applying these channels to the initially entangled state.

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



    # Prepare the initial entangled state |Φ⟩ = 1/√d * Σ |i⟩|i⟩
    dimension = 2 ** rails
    entangled_state = np.zeros((dimension, dimension), dtype=np.float64)
    for i in range(dimension):
        entangled_state[i, i] = 1
    entangled_state = entangled_state / np.sqrt(dimension)
    rho_initial = np.outer(entangled_state.flatten(), entangled_state.flatten().conj())

    # Kraus operators for the first channel
    K1 = generalized_amplitude_damping_channel(gamma_1, N_1)

    # Kraus operators for the second channel
    K2 = generalized_amplitude_damping_channel(gamma_2, N_2)

    # Apply the channels to the entangled state
    def apply_kraus_operators(ops, rho, dim):
        """Apply Kraus operators to a density matrix rho on a subsystem of given dimension."""
        new_rho = np.zeros_like(rho)
        for op in ops:
            # Construct the full operator to act on the entire system
            full_op = op
            for _ in range(rails - 1):
                full_op = kron(full_op, np.eye(2))
            new_rho += full_op @ rho @ full_op.conj().T
        return new_rho

    # Apply the channels sequentially to each rail
    rho_after_first = rho_initial
    for _ in range(rails):
        rho_after_first = apply_kraus_operators(K1, rho_after_first, dimension)

    rho_final = rho_after_first
    for _ in range(rails):
        rho_final = apply_kraus_operators(K2, rho_final, dimension)
    
    return rho_final

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('11.6', 3)
target = targets[0]

assert np.allclose(output_state(2,0,0,0,0), target)
target = targets[1]

assert np.allclose(output_state(2,1,0,1,0), target)
target = targets[2]

assert np.allclose(output_state(2,1,1,1,1), target)
