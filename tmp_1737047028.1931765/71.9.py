import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

# Background: In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is a column vector with a 1 in the j-th position and 0s elsewhere. 
# This is a standard basis vector in the context of quantum states. When dealing with multiple quantum systems, the tensor product of individual 
# kets is used to represent the combined state. The tensor product of vectors is a way to construct a new vector space from two or more vector spaces. 
# If j is a list, it represents multiple indices for which we need to create a tensor product of basis vectors. Similarly, if d is a list, it 
# represents the dimensions of each corresponding basis vector.


def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single dimension and single index
        out = np.zeros(dim)
        out[args] = 1.0
    else:
        # Multiple dimensions and indices
        vectors = []
        for d, j in zip(dim, args):
            vec = np.zeros(d)
            vec[j] = 1.0
            vectors.append(vec)
        # Compute the tensor product of all vectors
        out = vectors[0]
        for vec in vectors[1:]:
            out = np.kron(out, vec)
    
    return out


# Background: In linear algebra, the tensor product (also known as the Kronecker product) of two matrices is a way to construct a new matrix from two given matrices. 
# The tensor product of matrices is a generalization of the outer product of vectors. If A is an m×n matrix and B is a p×q matrix, 
# then their tensor product A ⊗ B is an mp×nq matrix. This operation is widely used in quantum mechanics to describe the state of a composite quantum system. 
# The tensor product of multiple matrices is computed by iteratively applying the Kronecker product to pairs of matrices.

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''

    
    if not args:
        raise ValueError("At least one matrix/vector is required for the tensor product.")
    
    # Start with the first matrix/vector
    M = args[0]
    
    # Iteratively compute the tensor product with the remaining matrices/vectors
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M


# Background: In quantum mechanics, a quantum channel is a mathematical model for noise and other processes affecting quantum states. 
# The action of a quantum channel on a quantum state can be described using Kraus operators. The Kraus representation of a quantum channel 
# is given by: $\mathcal{N}(\rho) = \sum_i K_i \rho K_i^\dagger$, where $K_i$ are the Kraus operators and $\rho$ is the density matrix of the state. 
# The Kraus operators satisfy the completeness relation $\sum_i K_i^\dagger K_i = \mathbb{I}$. When a channel acts on a specific subsystem of a 
# composite quantum state, the Kraus operators are applied to that subsystem while the identity operator acts on the other subsystems. 
# This is achieved by constructing the operator as a tensor product of identity operators and the Kraus operator, appropriately placed.


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
        result = np.zeros_like(rho)
        for Ki in K:
            result += Ki @ rho @ Ki.conj().T
        return result
    
    # Apply the channel to specified subsystems
    num_subsystems = len(dim)
    identity_operators = [np.eye(d) for d in dim]
    
    # Initialize the result density matrix
    result = np.zeros_like(rho)
    
    for Ki in K:
        # Construct the full operator with Ki acting on the specified subsystem
        operators = identity_operators.copy()
        for i, subsystem in enumerate(sys):
            operators[subsystem] = Ki if i == 0 else np.eye(dim[subsystem])
        
        # Compute the tensor product of the operators
        full_operator = operators[0]
        for op in operators[1:]:
            full_operator = np.kron(full_operator, op)
        
        # Apply the operator to the density matrix
        result += full_operator @ rho @ full_operator.conj().T
    
    return result


# Background: In quantum mechanics, a composite quantum system can be described by a density matrix that represents the state of the system. 
# This system can be composed of multiple subsystems, each with its own dimension. The order of these subsystems can be permuted, which 
# effectively changes the way the state is represented without altering the physical state itself. Permuting subsystems involves rearranging 
# the indices of the density matrix according to a specified permutation. This is important for tasks such as changing the basis of a 
# multipartite system or preparing the system for operations that require a specific subsystem order. The permutation is applied to both 
# the rows and columns of the density matrix, reflecting the new order of subsystems.



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
    # The first num_subsystems dimensions are for rows, the next num_subsystems are for columns
    reshaped_X = X.reshape(dim * 2)
    
    # Create the permutation for the tensor
    # We need to permute both the row and column indices
    permuted_indices = perm + [p + num_subsystems for p in perm]
    
    # Apply the permutation to the tensor
    permuted_tensor = np.transpose(reshaped_X, permuted_indices)
    
    # Reshape back to a 2D matrix
    Y = permuted_tensor.reshape((total_dim, total_dim))
    
    return Y


# Background: In quantum mechanics, the partial trace is an operation used to trace out (or discard) certain subsystems of a composite quantum system. 
# This is useful when we are interested in the state of a subsystem and want to ignore the rest. Mathematically, if a composite system is described 
# by a density matrix, the partial trace over a subsystem results in a reduced density matrix for the remaining subsystems. 
# For a system with subsystems of dimensions given by `dim`, the partial trace over specified subsystems involves summing over the degrees of freedom 
# of those subsystems. This can be achieved by reshaping the density matrix into a higher-dimensional tensor, permuting the axes to bring the 
# subsystems to be traced out together, and then summing over those axes.


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
    
    # Calculate the dimensions of the subsystems to keep
    dim_keep = [dim[i] for i in keep]
    
    # Calculate the dimensions of the subsystems to trace out
    dim_trace = [dim[i] for i in sys]
    
    # Reshape X into a tensor with shape (dim[0], dim[1], ..., dim[0], dim[1], ...)
    # The first num_subsystems dimensions are for rows, the next num_subsystems are for columns
    reshaped_X = X.reshape(dim * 2)
    
    # Create the permutation for the tensor
    # We need to permute both the row and column indices
    perm = keep + sys + [p + num_subsystems for p in keep] + [p + num_subsystems for p in sys]
    
    # Apply the permutation to the tensor
    permuted_tensor = np.transpose(reshaped_X, perm)
    
    # Reshape the permuted tensor to separate the traced and kept subsystems
    new_shape = dim_keep + dim_trace + dim_keep + dim_trace
    permuted_tensor = permuted_tensor.reshape(new_shape)
    
    # Sum over the traced out subsystems
    for i in range(len(sys)):
        permuted_tensor = np.trace(permuted_tensor, axis1=len(dim_keep), axis2=len(dim_keep) + len(dim_trace))
    
    # Reshape back to a 2D matrix
    result_dim = np.prod(dim_keep)
    Y = permuted_tensor.reshape((result_dim, result_dim))
    
    return Y


# Background: In quantum mechanics, the von Neumann entropy is a measure of the quantum information content of a quantum state, 
# represented by a density matrix ρ. It is the quantum analogue of the classical Shannon entropy. The von Neumann entropy is defined as:
# S(ρ) = -Tr(ρ log(ρ))
# where Tr denotes the trace operation, and log(ρ) is the matrix logarithm of ρ. The entropy quantifies the amount of uncertainty or 
# mixedness of the quantum state. A pure state has zero entropy, while a maximally mixed state has maximum entropy. 
# To compute the von Neumann entropy, we first need to find the eigenvalues of the density matrix ρ, as the entropy can be expressed 
# in terms of these eigenvalues: S(ρ) = -Σ λ_i log(λ_i), where λ_i are the eigenvalues of ρ.



def entropy(rho):
    '''Inputs:
    rho: 2d array of floats with equal dimensions, the density matrix of the state
    Output:
    en: quantum (von Neumann) entropy of the state rho, float
    '''
    # Compute the eigenvalues of the density matrix
    eigenvalues = np.linalg.eigvalsh(rho)
    
    # Filter out zero eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    # Calculate the von Neumann entropy
    en = -np.sum(eigenvalues * np.log(eigenvalues))
    
    return en


# Background: In quantum mechanics, the generalized amplitude damping channel (GADC) is a model that describes the interaction of a quantum system with a thermal bath at a finite temperature. It is an extension of the amplitude damping channel, which models energy dissipation. The GADC is characterized by two parameters: the damping parameter γ, which represents the probability of energy loss, and the thermal parameter N, which represents the mean number of excitations in the environment. The channel is described by four Kraus operators, which are 2x2 matrices that act on a single qubit. These operators are used to express the effect of the channel on a quantum state.


def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''
    # Define the Kraus operators for the generalized amplitude damping channel
    K1 = np.array([[np.sqrt(1 - N), 0],
                   [0, np.sqrt((1 - N) * (1 - gamma))]])
    
    K2 = np.array([[0, np.sqrt(gamma * (1 - N))],
                   [0, 0]])
    
    K3 = np.array([[np.sqrt(N * (1 - gamma)), 0],
                   [0, np.sqrt(N)]])
    
    K4 = np.array([[0, 0],
                   [np.sqrt(gamma * N), 0]])
    
    # Return the list of Kraus operators
    kraus = [K1, K2, K3, K4]
    
    return kraus


# Background: The reverse coherent information of a bipartite quantum state ρ is a measure of the quantum correlations between two subsystems A and B. 
# It is defined as I_R(A⟩B) = S(A)_ρ - S(AB)_ρ, where S(X)_ρ is the von Neumann entropy of the reduced state on subsystem X. 
# The von Neumann entropy S(ρ) is calculated as S(ρ) = -Tr(ρ log(ρ)), where Tr denotes the trace operation and log(ρ) is the matrix logarithm of ρ. 
# In this problem, we consider a bipartite state |ψ⟩ = √(1-p)|00⟩ + √p|11⟩, where one qubit is sent through a generalized amplitude damping channel (GADC) 
# characterized by parameters γ and N. The task is to compute the negative of the reverse coherent information of the output state after the channel.



def neg_rev_coh_info(p, g, N):
    '''Calculates the negative of coherent information of the output state
    Inputs:
    p: float, parameter for the input state
    g: float, damping parameter
    N: float, thermal parameter
    Outputs:
    neg_I_c: float, negative of coherent information of the output state
    '''
    
    # Define the initial state |ψ⟩ = √(1-p)|00⟩ + √p|11⟩
    psi = np.array([np.sqrt(1 - p), 0, 0, np.sqrt(p)])
    rho = np.outer(psi, psi.conj())
    
    # Get the Kraus operators for the generalized amplitude damping channel
    K1 = np.array([[np.sqrt(1 - N), 0],
                   [0, np.sqrt((1 - N) * (1 - g))]])
    
    K2 = np.array([[0, np.sqrt(g * (1 - N))],
                   [0, 0]])
    
    K3 = np.array([[np.sqrt(N * (1 - g)), 0],
                   [0, np.sqrt(N)]])
    
    K4 = np.array([[0, 0],
                   [np.sqrt(g * N), 0]])
    
    kraus = [K1, K2, K3, K4]
    
    # Apply the channel to the second qubit
    output_rho = np.zeros_like(rho)
    for K in kraus:
        K_full = np.kron(np.eye(2), K)
        output_rho += K_full @ rho @ K_full.conj().T
    
    # Calculate the reduced density matrix for subsystem A
    dim = [2, 2]  # Dimensions of the subsystems
    sys = [1]  # Trace out the second subsystem
    reduced_rho_A = partial_trace(output_rho, sys, dim)
    
    # Calculate the entropies
    S_AB = entropy(output_rho)
    S_A = entropy(reduced_rho_A)
    
    # Calculate the negative of the reverse coherent information
    neg_I_R = S_AB - S_A
    
    return neg_I_R

def partial_trace(X, sys, dim):
    '''Helper function to calculate the partial trace over specified subsystems.'''
    num_subsystems = len(dim)
    keep = [i for i in range(num_subsystems) if i not in sys]
    dim_keep = [dim[i] for i in keep]
    dim_trace = [dim[i] for i in sys]
    
    reshaped_X = X.reshape(dim * 2)
    perm = keep + sys + [p + num_subsystems for p in keep] + [p + num_subsystems for p in sys]
    permuted_tensor = np.transpose(reshaped_X, perm)
    
    new_shape = dim_keep + dim_trace + dim_keep + dim_trace
    permuted_tensor = permuted_tensor.reshape(new_shape)
    
    for i in range(len(sys)):
        permuted_tensor = np.trace(permuted_tensor, axis1=len(dim_keep), axis2=len(dim_keep) + len(dim_trace))
    
    result_dim = np.prod(dim_keep)
    Y = permuted_tensor.reshape((result_dim, result_dim))
    
    return Y

def entropy(rho):
    '''Helper function to calculate the von Neumann entropy of a density matrix.'''
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 0]
    en = -np.sum(eigenvalues * np.log(eigenvalues))
    return en



# Background: In quantum information theory, the coherent information of a quantum channel is a measure of the channel's ability to preserve quantum information. 
# For a generalized amplitude damping channel (GADC), the reverse coherent information is defined as the maximum reverse coherent information over all possible 
# input pure states. The reverse coherent information for a bipartite state ρ is given by I_R(A⟩B) = S(A)_ρ - S(AB)_ρ, where S(X)_ρ is the von Neumann entropy 
# of the reduced state on subsystem X. To find the channel coherent information, we need to maximize this quantity over all pure input states of the form 
# |ψ⟩ = √p|00⟩ + √(1-p)|11⟩. This involves calculating the reverse coherent information for each state and finding the maximum value.

def GADC_rev_coh_inf(g, N):
    '''Calculates the coherent information of the GADC.
    Inputs:
    g: float, damping parameter
    N: float, thermal parameter
    Outputs:
    channel_coh_info: float, channel coherent information of a GADC
    '''
    def neg_rev_coh_info(p):
        # Define the initial state |ψ⟩ = √(1-p)|00⟩ + √p|11⟩
        psi = np.array([np.sqrt(1 - p), 0, 0, np.sqrt(p)])
        rho = np.outer(psi, psi.conj())
        
        # Get the Kraus operators for the generalized amplitude damping channel
        K1 = np.array([[np.sqrt(1 - N), 0],
                       [0, np.sqrt((1 - N) * (1 - g))]])
        
        K2 = np.array([[0, np.sqrt(g * (1 - N))],
                       [0, 0]])
        
        K3 = np.array([[np.sqrt(N * (1 - g)), 0],
                       [0, np.sqrt(N)]])
        
        K4 = np.array([[0, 0],
                       [np.sqrt(g * N), 0]])
        
        kraus = [K1, K2, K3, K4]
        
        # Apply the channel to the second qubit
        output_rho = np.zeros_like(rho)
        for K in kraus:
            K_full = np.kron(np.eye(2), K)
            output_rho += K_full @ rho @ K_full.conj().T
        
        # Calculate the reduced density matrix for subsystem A
        dim = [2, 2]  # Dimensions of the subsystems
        sys = [1]  # Trace out the second subsystem
        reduced_rho_A = partial_trace(output_rho, sys, dim)
        
        # Calculate the entropies
        S_AB = entropy(output_rho)
        S_A = entropy(reduced_rho_A)
        
        # Calculate the negative of the reverse coherent information
        neg_I_R = S_AB - S_A
        
        return neg_I_R

    # Use fminbound to find the p that minimizes the negative reverse coherent information
    p_opt = fminbound(neg_rev_coh_info, 0, 1)
    
    # Calculate the channel coherent information
    channel_rev_coh_info = -neg_rev_coh_info(p_opt)
    
    return channel_rev_coh_info

def partial_trace(X, sys, dim):
    '''Helper function to calculate the partial trace over specified subsystems.'''
    num_subsystems = len(dim)
    keep = [i for i in range(num_subsystems) if i not in sys]
    dim_keep = [dim[i] for i in keep]
    dim_trace = [dim[i] for i in sys]
    
    reshaped_X = X.reshape(dim * 2)
    perm = keep + sys + [p + num_subsystems for p in keep] + [p + num_subsystems for p in sys]
    permuted_tensor = np.transpose(reshaped_X, perm)
    
    new_shape = dim_keep + dim_trace + dim_keep + dim_trace
    permuted_tensor = permuted_tensor.reshape(new_shape)
    
    for i in range(len(sys)):
        permuted_tensor = np.trace(permuted_tensor, axis1=len(dim_keep), axis2=len(dim_keep) + len(dim_trace))
    
    result_dim = np.prod(dim_keep)
    Y = permuted_tensor.reshape((result_dim, result_dim))
    
    return Y

def entropy(rho):
    '''Helper function to calculate the von Neumann entropy of a density matrix.'''
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 0]
    en = -np.sum(eigenvalues * np.log(eigenvalues))
    return en


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('71.9', 5)
target = targets[0]

assert np.allclose(GADC_rev_coh_inf(0.2,0.4), target)
target = targets[1]

assert np.allclose(GADC_rev_coh_inf(0.2,0.1), target)
target = targets[2]

assert np.allclose(GADC_rev_coh_inf(0.4,0.2), target)
target = targets[3]

assert np.allclose(GADC_rev_coh_inf(0,0), target)
target = targets[4]

assert np.allclose(GADC_rev_coh_inf(1,1), target)
