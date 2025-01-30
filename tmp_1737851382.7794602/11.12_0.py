import numpy as np
import itertools
import scipy.linalg

# Background: In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is a column vector with a 1 in the j-th position and 0s elsewhere. 
# This is known as a standard basis vector. When dealing with multiple quantum systems, the state of the combined system is represented by the 
# tensor product of the individual states. If j is a list, it represents multiple indices for which we need to create a tensor product of 
# standard basis vectors. If d is a list, it specifies the dimensions of each individual space for the tensor product.




def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single space case
        d = dim
        j = args
        if not isinstance(j, int):
            raise ValueError("Index must be an integer for single dimension.")
        if j < 0 or j >= d:
            raise IndexError("Index out of bounds for the dimension specified.")
        # Create a d-dimensional basis vector with 1 at position j
        out = np.zeros(d)
        out[j] = 1.0
    else:
        # Multiple spaces case
        d_list = dim
        j_list = args
        if not isinstance(j_list, list):
            raise ValueError("Indices must be a list for multiple dimensions.")
        if len(d_list) != len(j_list):
            raise ValueError("Dimensions and indices lengths do not match.")
        # Create the tensor product of basis vectors
        basis_vectors = []
        for d, j in zip(d_list, j_list):
            if j < 0 or j >= d:
                raise IndexError("Index out of bounds for the dimension specified.")
            vec = np.zeros(d)
            vec[j] = 1.0
            basis_vectors.append(vec)
        # Compute the tensor product
        out = basis_vectors[0]
        for vec in basis_vectors[1:]:
            out = np.kron(out, vec)
    
    return out


# Background: In quantum mechanics, a bipartite maximally entangled state is a quantum state of two subsystems that cannot be described independently.
# One common example is the Bell state, which is a specific type of entangled state. In the context of multi-rail encoding, each party's state is
# represented across multiple "rails" or dimensions. The goal is to create a density matrix representing a maximally entangled state where each
# subsystem is encoded using m rails. The density matrix is a square matrix that describes the statistical state of a quantum system, and for a
# maximally entangled state, it will have specific properties that reflect the entanglement between the subsystems.




def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    if not isinstance(rails, int):
        raise TypeError("Rails must be an integer.")
    if rails < 0:
        raise ValueError("Rails must be a non-negative integer.")
    
    # Total dimension for each subsystem
    dim = 2 ** rails
    
    # Create the maximally entangled state |Φ⟩ = (1/√dim) * Σ |i⟩|i⟩
    # where i ranges over all possible states of the subsystem
    state_vector = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        # Create the basis vector |i⟩ for the first subsystem
        ket_i = np.zeros(dim, dtype=np.complex128)
        ket_i[i] = 1.0
        
        # Create the basis vector |i⟩ for the second subsystem
        ket_j = np.zeros(dim, dtype=np.complex128)
        ket_j[i] = 1.0
        
        # Tensor product |i⟩|i⟩
        state_vector += np.outer(ket_i, ket_j)
    
    # Normalize the state vector
    state_vector /= np.sqrt(dim)
    
    # Create the density matrix ρ = |Φ⟩⟨Φ|
    state = np.outer(state_vector.flatten(), state_vector.flatten().conj())
    
    # Ensure the matrix is Hermitian and has non-negative eigenvalues
    state = (state + state.conj().T) / 2
    
    return state


# Background: In linear algebra and quantum mechanics, the tensor product (also known as the Kronecker product) is an operation on two matrices or vectors
# that results in a block matrix. For vectors, it results in a higher-dimensional vector. The tensor product of matrices A and B, denoted A ⊗ B, is a 
# matrix formed by multiplying each element of A by the entire matrix B. This operation is associative, meaning the order of operations does not affect 
# the result, allowing us to compute the tensor product of an arbitrary number of matrices or vectors sequentially.

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''


    if len(args) == 0:
        raise ValueError("At least one matrix or vector is required for the tensor product.")

    # Start with the first matrix/vector
    try:
        M = args[0].astype(np.float64)  # Ensure the first matrix/vector is of type float64
    except ValueError as e:
        raise TypeError("Input matrices must contain numeric values.") from e

    # Sequentially compute the tensor product with each subsequent matrix/vector
    for matrix in args[1:]:
        try:
            M = np.kron(M, matrix.astype(np.float64))  # Convert each matrix to float64 before the operation
        except ValueError as e:
            raise TypeError("Input matrices must contain numeric values.") from e

    return M


# Background: In quantum mechanics, a quantum channel is a mathematical model for the physical process of transmitting quantum states. 
# It is represented by a set of Kraus operators {K_i}, which are matrices that describe the effect of the channel on a quantum state. 
# The action of a quantum channel on a density matrix ρ is given by the transformation ρ' = Σ_i K_i ρ K_i†, where K_i† is the conjugate 
# transpose of K_i. When applying a channel to specific subsystems of a composite quantum system, we need to consider the tensor product 
# structure of the state and apply the channel only to the specified subsystems. This involves using the tensor product to construct the 
# full operator that acts on the entire system, while the identity operator acts on the subsystems not affected by the channel.

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




    if not K:
        raise ValueError("Kraus operators list is empty")

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Input density matrix rho must be square")

    if (sys is None) != (dim is None):
        raise ValueError("Both sys and dim must be provided together or both set to None")

    if sys is not None:
        if any(s >= len(dim) for s in sys):
            raise IndexError("Subsystem index out of range")

    if sys is None or dim is None:
        # Apply the channel to the entire system
        new_rho = np.zeros_like(rho, dtype=np.complex128)
        for K_i in K:
            new_rho += K_i @ rho @ K_i.conj().T
        return new_rho

    # Apply the channel to specified subsystems
    total_dim = np.prod(dim)
    identity = np.eye(total_dim, dtype=np.complex128)
    new_rho = np.zeros((total_dim, total_dim), dtype=np.complex128)

    # Generate all possible indices for the subsystems
    indices = list(itertools.product(*[range(d) for d in dim]))

    for idx in indices:
        # Create the basis vector |idx⟩
        ket_idx = np.zeros(total_dim, dtype=np.complex128)
        ket_idx[np.ravel_multi_index(idx, dim)] = 1.0

        # Create the projector |idx⟩⟨idx|
        projector = np.outer(ket_idx, ket_idx.conj())

        # Apply the channel to the specified subsystems
        for K_i in K:
            # Construct the full operator that acts on the entire system
            full_operator = np.eye(1, dtype=np.complex128)
            for s, d in enumerate(dim):
                # Apply the Kraus operator to the subsystem
                if s in sys:
                    full_operator = np.kron(full_operator, K_i)
                else:
                    full_operator = np.kron(full_operator, np.eye(d, dtype=np.complex128))

            # Update the new density matrix
            new_rho += full_operator @ rho @ full_operator.conj().T

    return new_rho



# Background: In quantum mechanics, the generalized amplitude damping channel is a model that describes the interaction of a quantum system with a thermal bath at a finite temperature. It is characterized by two parameters: the damping parameter γ, which represents the probability of energy dissipation, and the thermal parameter N, which represents the average number of excitations in the environment. The channel is described by a set of Kraus operators {A1, A2, A3, A4}, which are 2x2 matrices that transform the density matrix of the quantum system. These operators are used to model the effects of the environment on the quantum state, including both energy loss and thermal noise.

def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''
    if gamma < 0 or gamma > 1 or N < 0:
        raise ValueError("Invalid parameters: gamma must be between 0 and 1, N must be non-negative.")

    # Calculate the probability of the system being in the excited state
    p = N / (N + 1)
    
    # Define the Kraus operators
    A1 = np.array([[np.sqrt(p * (1 - gamma)), 0],
                   [0, np.sqrt(1 - gamma)]], dtype=np.float64)
    
    A2 = np.array([[0, np.sqrt(gamma * (1 - p))],
                   [0, 0]], dtype=np.float64)
    
    A3 = np.array([[np.sqrt((1 - p) * gamma), 0],
                   [0, 0]], dtype=np.float64)
    
    A4 = np.array([[0, 0],
                   [np.sqrt(p * gamma), np.sqrt(p)]], dtype=np.float64)
    
    # Return the list of Kraus operators
    return [A1, A2, A3, A4]


# Background: In quantum information theory, when a quantum state is transmitted through a noisy channel, it undergoes transformations
# described by the channel's Kraus operators. For a multi-rail encoded state, each rail (or subsystem) can be affected by a different
# channel. The generalized amplitude damping channel models the interaction of a quantum system with a thermal environment, characterized
# by parameters gamma (damping) and N (thermal noise). To find the output state after transmission, we apply the Kraus operators of the
# channel to each rail of the state. This involves constructing the tensor product of the Kraus operators for each rail and applying them
# to the density matrix of the state.

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



    # Validate input parameters
    if not isinstance(rails, int) or rails < 0:
        raise ValueError("Rails must be a non-negative integer.")
    if not (0 <= gamma_1 <= 1 and 0 <= gamma_2 <= 1):
        raise ValueError("Gamma values must be between 0 and 1.")
    if not (0 <= N_1 <= 1 and 0 <= N_2 <= 1):
        raise ValueError("N values must be between 0 and 1.")
    if not isinstance(gamma_1, (int, float)) or not isinstance(gamma_2, (int, float)):
        raise TypeError("Gamma values must be real numbers.")
    if not isinstance(N_1, (int, float)) or not isinstance(N_2, (int, float)):
        raise TypeError("N values must be real numbers.")

    # Generate the initial maximally entangled state
    dim = 2 ** rails
    state_vector = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        ket_i = np.zeros(dim, dtype=np.complex128)
        ket_i[i] = 1.0
        state_vector += np.outer(ket_i, ket_i)
    state_vector /= np.sqrt(dim)
    rho = np.outer(state_vector.flatten(), state_vector.flatten().conj())

    # Get the Kraus operators for both channels
    K1 = generalized_amplitude_damping_channel(gamma_1, N_1)
    K2 = generalized_amplitude_damping_channel(gamma_2, N_2)

    # Apply the channels to each rail
    for rail in range(rails):
        # Apply channel 1 to the first subsystem of each rail
        rho = apply_channel(K1, rho, sys=[rail], dim=[2] * (2 * rails))
        # Apply channel 2 to the second subsystem of each rail
        rho = apply_channel(K2, rho, sys=[rail + rails], dim=[2] * (2 * rails))

    return rho




# Background: In quantum mechanics, a measurement projector is an operator that projects a quantum state onto a specific subspace. 
# For a system of qubits, the one-particle sector refers to the subspace where exactly one qubit is in the state |1⟩ and the rest are in |0⟩. 
# In the context of multi-rail encoding, each rail represents a qubit, and we are interested in the subspace where each receiver has exactly 
# one qubit in the state |1⟩ across their respective rails. The global projector for this measurement is constructed by considering all 
# possible combinations of one |1⟩ and (m-1) |0⟩ states for each set of m rails, and then taking the tensor product of these projectors 
# for both receivers.

def measurement(rails):
    '''Returns the measurement projector
    Input:
    rails: int, number of rails
    Output:
    global_proj: (2**(2*rails), 2**(2*rails)) dimensional array of floats
    '''
    if not isinstance(rails, int):
        raise TypeError("rails must be an integer")
    if rails < 0:
        raise ValueError("rails must be non-negative")
    
    # Total dimension for each subsystem
    dim = 2 ** rails

    # Initialize the global projector as a zero matrix
    global_proj = np.zeros((dim**2, dim**2), dtype=np.float64)

    # Generate all possible indices for the one-particle sector
    indices = list(itertools.combinations(range(rails), 1))

    # Iterate over all combinations of one-particle states for both subsystems
    for idx1 in indices:
        for idx2 in indices:
            # Create the basis vector |idx1⟩ for the first subsystem
            ket1 = np.zeros(dim, dtype=np.float64)
            ket1[idx1[0]] = 1.0

            # Create the basis vector |idx2⟩ for the second subsystem
            ket2 = np.zeros(dim, dtype=np.float64)
            ket2[idx2[0]] = 1.0

            # Create the projector |idx1⟩⟨idx1| ⊗ |idx2⟩⟨idx2|
            proj1 = np.outer(ket1, ket1)
            proj2 = np.outer(ket2, ket2)
            proj = np.kron(proj1, proj2)

            # Add the projector to the global projector
            global_proj += proj

    # Handle the special case when rails = 0
    if rails == 0:
        global_proj = np.array([[1.0]])

    return global_proj


# Background: In quantum mechanics, a multipartite quantum state can be represented by a density matrix that describes the state of multiple subsystems. 
# The order of these subsystems can be permuted to reflect different arrangements or interactions between them. Permuting the subsystems involves rearranging 
# the dimensions of the density matrix according to a specified order. This is achieved by reshaping the matrix into a higher-dimensional tensor, 
# permuting the axes of this tensor, and then flattening it back into a matrix. The permutation of subsystems is crucial in quantum information processing 
# tasks where the order of operations or interactions between subsystems matters.

def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    perm: list of int containing the desired order
    dim: list of int containing the dimensions of all subsystems.
    Output:
    Y: 2d array of floats with equal dimensions, the density matrix of the permuted state
    '''


    # Validate input dimensions
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("Input X must be a square matrix.")
    if len(dim) != len(perm):
        raise ValueError("Dimension list and permutation list must have the same length.")
    if sorted(perm) != list(range(len(dim))):
        raise ValueError("Permutation list must be a valid permutation of subsystem indices.")

    # Calculate the total dimension
    total_dim = np.prod(dim)
    if X.shape[0] != total_dim:
        raise ValueError("The dimensions of X do not match the product of the dimensions in dim.")

    # Reshape X into a tensor with shape (dim[0], dim[1], ..., dim[0], dim[1], ...)
    reshaped_X = X.reshape(tuple(dim) * 2)

    # Create the permutation for the axes
    num_subsystems = len(dim)
    permuted_axes = [perm[i] for i in range(num_subsystems)] + [perm[i] + num_subsystems for i in range(num_subsystems)]

    # Permute the axes of the tensor
    permuted_tensor = np.transpose(reshaped_X, axes=permuted_axes)

    # Reshape back into a matrix
    Y = permuted_tensor.reshape(total_dim, total_dim)

    return Y


# Background: In quantum mechanics, the partial trace is an operation used to trace out (or discard) certain subsystems of a composite quantum system. 
# This operation is crucial when we are interested in the state of a subsystem, rather than the entire system. The partial trace over a subsystem 
# effectively reduces the dimensionality of the density matrix by summing over the degrees of freedom of the traced-out subsystem. 
# Mathematically, if we have a composite system described by a density matrix ρ, and we want to trace out subsystem B, the resulting density matrix 
# for subsystem A is obtained by summing over the indices corresponding to subsystem B. The syspermute function can be used to rearrange the 
# subsystems so that the ones to be traced out are contiguous, simplifying the partial trace operation.


def syspermute(X, perm, dim):
    '''Permute the dimensions of the density matrix according to the permutation list.'''
    assert len(dim) == len(perm), "Dimension list and permutation list must be of the same length."
    total_dim = np.prod(dim)
    shape = [dim[i] for i in perm] + [dim[i] for i in perm]
    X_perm = X.reshape(shape)
    X_perm = np.transpose(X_perm, perm + [len(dim) + i for i in perm])
    return X_perm.reshape(total_dim, total_dim)

def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''
    # Validate input dimensions
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("Input X must be a square matrix.")
    if any(s >= len(dim) for s in sys):
        raise IndexError("Subsystem index out of range.")

    # Calculate the total dimension
    total_dim = np.prod(dim)
    if X.shape[0] != total_dim:
        raise ValueError("The dimensions of X do not match the product of the dimensions in dim.")

    # Determine the dimensions of the subsystems to keep
    keep = [i for i in range(len(dim)) if i not in sys]
    dim_keep = [dim[i] for i in keep]
    dim_trace = [dim[i] for i in sys]

    # Permute the subsystems so that the ones to be traced out are at the end
    perm = keep + sys
    X_permuted = syspermute(X, perm, dim)

    # Reshape the permuted matrix into a tensor
    reshaped_X = X_permuted.reshape([dim[i] for i in perm] * 2)

    # Perform the partial trace over the specified subsystems
    for i in range(len(sys)):
        reshaped_X = np.trace(reshaped_X, axis1=len(keep) + i, axis2=len(keep) + len(sys) + i)

    # Reshape back into a matrix
    result_dim = np.prod(dim_keep)
    Y = reshaped_X.reshape(result_dim, result_dim)

    return Y


# Background: The von Neumann entropy is a measure of the quantum uncertainty or disorder of a quantum state, analogous to the classical Shannon entropy. 
# It is defined for a quantum state represented by a density matrix ρ as S(ρ) = -Tr(ρ log2 ρ), where Tr denotes the trace operation. 
# The entropy quantifies the amount of quantum information or the degree of mixedness of the state. 
# For pure states, the von Neumann entropy is zero, while for mixed states, it is positive. 
# To compute the entropy, we first need to find the eigenvalues of the density matrix, as the entropy is calculated using these eigenvalues.



def entropy(rho):
    '''Inputs:
    rho: 2d array of floats with equal dimensions, the density matrix of the state
    Output:
    en: quantum (von Neumann) entropy of the state rho, float
    '''
    # Validate input
    if not isinstance(rho, np.ndarray):
        raise TypeError("Input rho must be a numpy array.")
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Input rho must be a square matrix.")
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("Input rho must be Hermitian.")
    if not np.isclose(np.trace(rho), 1):
        raise ValueError("Trace of rho must be 1.")

    # Compute the eigenvalues of the density matrix
    eigenvalues = scipy.linalg.eigvalsh(rho)

    # Check for negative eigenvalues which are not allowed in a valid density matrix
    if np.any(eigenvalues < 0):
        raise ValueError("Density matrix cannot have negative eigenvalues.")

    # Filter out zero and negative eigenvalues to avoid log(0) and log of negative
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Calculate the von Neumann entropy
    en = -np.sum(eigenvalues * np.log2(eigenvalues))

    return en


# Background: Coherent information is a measure of the quantum information that can be transmitted from one part of a bipartite quantum system to another. 
# It is defined for a bipartite state ρ_AB as I(A⟩B) = S(ρ_B) - S(ρ_AB), where S(ρ) is the von Neumann entropy of the state ρ, ρ_B is the reduced density 
# matrix of system B obtained by tracing out system A, and ρ_AB is the joint state of systems A and B. The coherent information quantifies the amount of 
# quantum information that can be preserved when system A is sent through a quantum channel to system B. It is a key concept in quantum information theory, 
# particularly in the context of quantum error correction and quantum communication.


def entropy(rho):
    """Calculate the von Neumann entropy of a density matrix."""
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter out zero eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 0]
    return -np.sum(eigenvalues * np.log2(eigenvalues))

def partial_trace(rho, keep, dims):
    """Perform a partial trace."""
    tr_shape = np.array(dims)
    keep = np.array(keep)
    tr_axes = tuple(np.delete(np.arange(len(dims)), keep))
    new_shape = np.prod(tr_shape[keep]), np.prod(tr_shape[tr_axes]), tr_shape[keep], tr_shape[tr_axes]
    rho = rho.reshape(new_shape)
    return np.trace(rho, axis1=1, axis2=3)

def coherent_inf_state(rho_AB, dimA, dimB):
    '''Inputs:
    rho_AB: 2d array of floats with equal dimensions, the state we evaluate coherent information
    dimA: int, dimension of system A
    dimB: int, dimension of system B
    Output
    co_inf: float, the coherent information of the state rho_AB
    '''
    # Validate input
    if not isinstance(rho_AB, np.ndarray):
        raise TypeError("rho_AB must be a numpy array.")
    if not isinstance(dimA, int) or not isinstance(dimB, int):
        raise TypeError("dimA and dimB must be integers.")
    if dimA <= 0 or dimB <= 0:
        raise ValueError("Dimensions must be positive integers.")
    if rho_AB.ndim != 2 or rho_AB.shape[0] != rho_AB.shape[1]:
        raise ValueError("rho_AB must be a square matrix.")
    if not np.allclose(rho_AB, rho_AB.conj().T):
        raise ValueError("rho_AB must be Hermitian.")
    if not np.isclose(np.trace(rho_AB), 1):
        raise ValueError("Trace of rho_AB must be 1.")
    if dimA * dimB != rho_AB.shape[0]:
        raise ValueError("The dimensions of A and B do not match the size of rho_AB.")

    # Calculate the reduced density matrix rho_B by tracing out system A
    dim = [dimA, dimB]
    rho_B = partial_trace(rho_AB, [1], dim)

    # Calculate the von Neumann entropy of rho_AB
    S_AB = entropy(rho_AB)

    # Calculate the von Neumann entropy of rho_B
    S_B = entropy(rho_B)

    # Calculate the coherent information
    co_inf = S_B - S_AB

    return co_inf



# Background: In quantum information theory, the hashing protocol is a method used to distill entanglement from a set of quantum states. 
# The rate of entanglement production is determined by the coherent information of the state after certain operations, such as measurements 
# and post-selection. The coherent information quantifies the amount of quantum information that can be transmitted through a quantum channel. 
# In this context, we are interested in calculating the rate of entanglement per channel, which is given by the coherent information of the 
# post-selected state. This involves computing the coherent information of the state after it has been affected by generalized amplitude 
# damping channels and measured in the one-particle sector.




def rate(rails, gamma_1, N_1, gamma_2, N_2):
    '''Inputs:
    rails: int, number of rails
    gamma_1: float, damping parameter of the first channel
    N_1: float, thermal parameter of the first channel
    gamma_2: float, damping parameter of the second channel
    N_2: float, thermal parameter of the second channel
    Output: float, the achievable rate of our protocol
    '''
    # Generate the output state after passing through the channels
    rho_out = output_state(rails, gamma_1, N_1, gamma_2, N_2)

    # Get the measurement projector for the one-particle sector
    projector = measurement(rails)

    # Post-select the state using the projector
    post_selected_state = projector @ rho_out @ projector
    post_selected_state /= np.trace(post_selected_state)  # Normalize the post-selected state

    # Calculate the dimensions of the subsystems
    dimA = 2 ** rails
    dimB = 2 ** rails

    # Calculate the coherent information of the post-selected state
    rate = coherent_inf_state(post_selected_state, dimA, dimB)

    return rate

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('11.12', 5)
target = targets[0]

assert np.allclose(rate(2,0.2,0.2,0.2,0.2), target)
target = targets[1]

assert np.allclose(rate(2,0.3,0.4,0.2,0.2), target)
target = targets[2]

assert np.allclose(rate(3,0.4,0.1,0.1,0.2), target)
target = targets[3]

assert np.allclose(rate(2,0,0,0,0), target)
target = targets[4]

assert np.allclose(rate(2,0.2,0,0.4,0), target)
