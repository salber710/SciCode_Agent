import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

# Background: In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is a column vector with a 1 in the j-th position and 0s elsewhere. 
# This is a standard basis vector in the context of quantum states. When dealing with multiple quantum systems, the tensor product of individual 
# kets is used to represent the combined state. The tensor product of vectors results in a higher-dimensional vector space, where the dimensions 
# are the product of the individual dimensions. In this problem, we need to construct such a ket vector or a tensor product of multiple ket vectors 
# based on the input dimensions and indices.


def ket(dim, *args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single dimension and single index
        if not isinstance(args[0], int):
            raise TypeError("For single dimension, index must be an integer.")
        j = args[0]
        if j < 0 or j >= dim:
            raise IndexError("Index out of bounds.")
        out = np.zeros(dim)
        out[j] = 1.0
    elif isinstance(dim, list) and isinstance(args[0], list):
        # Multiple dimensions and multiple indices
        dims = dim
        indices = args[0]
        if len(dims) != len(indices):
            raise ValueError("Dimensions and indices must have the same length.")
        if not dims or not indices:
            raise ValueError("Dimensions and indices cannot be empty.")
        # Validate dimensions are positive and indices are within bounds
        for d, idx in zip(dims, indices):
            if d <= 0:
                raise ValueError("Dimensions must be positive integers.")
            if idx < 0 or idx >= d:
                raise IndexError("Index out of bounds.")
        # Start with the first ket
        out = np.zeros(dims[0])
        out[indices[0]] = 1.0
        # Tensor product with subsequent kets
        for d, j in zip(dims[1:], indices[1:]):
            ket_j = np.zeros(d)
            ket_j[j] = 1.0
            out = np.kron(out, ket_j)
    else:
        raise ValueError("Invalid input format for dim and args.")
    
    return out


# Background: In linear algebra and quantum mechanics, the tensor product (also known as the Kronecker product) is an operation on two matrices or vectors that results in a block matrix. For vectors, the tensor product results in a higher-dimensional vector. For matrices, it results in a larger matrix that combines the information of the input matrices. The tensor product is essential in quantum mechanics for describing the state of a composite quantum system. The Kronecker product of matrices A (of size m x n) and B (of size p x q) is a matrix of size (m*p) x (n*q).


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one input matrix/vector is required.")
    
    # Start with the first matrix/vector
    M = args[0]
    
    # Iterate over the remaining matrices/vectors and compute the Kronecker product
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M


# Background: In quantum mechanics, a quantum channel is a mathematical model for the physical process of transmitting quantum states. 
# The action of a quantum channel on a quantum state can be described using Kraus operators. The Kraus representation of a quantum channel 
# is given by the equation: 𝒩(ρ) = ∑_i K_i ρ K_i^†, where K_i are the Kraus operators and K_i^† is the conjugate transpose of K_i. 
# The Kraus operators satisfy the completeness relation ∑_i K_i^† K_i = I, where I is the identity operator. 
# When a quantum channel acts on a specific subsystem of a composite quantum state, the Kraus operators are applied to that subsystem, 
# while the identity operator acts on the other subsystems. This is achieved by taking the tensor product of the identity operators 
# and the Kraus operators, ensuring that the channel acts only on the specified subsystem.



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
        dim = [rho.shape[0]]
        sys = [0]

    # Calculate the total dimension of the system
    total_dim = np.prod(dim)

    # Initialize the output density matrix
    output_rho = np.zeros((total_dim, total_dim), dtype=complex)

    # Iterate over each Kraus operator
    for K_i in K:
        # Construct the full operator for the system
        full_operator = 1
        for i in range(len(dim)):
            if i in sys:
                # Apply the Kraus operator to the specified subsystem
                full_operator = np.kron(full_operator, K_i)
            else:
                # Apply the identity operator to the other subsystems
                full_operator = np.kron(full_operator, np.eye(dim[i]))

        # Apply the channel to the state rho
        output_rho += full_operator @ rho @ full_operator.conj().T

    return output_rho


# Background: In quantum mechanics, a composite quantum system can be described by a density matrix that represents the state of the system. 
# This system can be composed of multiple subsystems, each with its own dimension. Sometimes, it is necessary to permute the order of these 
# subsystems, which involves rearranging the dimensions of the density matrix according to a specified permutation. This operation is crucial 
# for tasks such as changing the basis of a quantum state or preparing a state for a specific quantum operation. The permutation of subsystems 
# is achieved by reshaping the density matrix into a multi-dimensional array, permuting the axes according to the desired order, and then 
# reshaping it back into a matrix form.


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
    
    # Ensure the permutation is valid
    if sorted(perm) != list(range(num_subsystems)):
        raise ValueError("Invalid permutation order.")
    
    # Check for non-integer elements in perm and dim
    if not all(isinstance(p, int) for p in perm):
        raise TypeError("All elements in perm must be integers.")
    if not all(isinstance(d, int) for d in dim):
        raise TypeError("All elements in dim must be integers.")
    
    # Check for negative or zero dimensions
    if any(d <= 0 for d in dim):
        raise ValueError("Dimensions must be positive integers.")
    
    # Reshape X into a multi-dimensional array
    try:
        reshaped_X = np.reshape(X, dim + dim)
    except ValueError:
        raise ValueError("Cannot reshape array to these dimensions.")
    
    # Create the permutation for the axes
    permuted_axes = perm + [p + num_subsystems for p in perm]
    
    # Permute the axes of the reshaped array
    permuted_X = np.transpose(reshaped_X, permuted_axes)
    
    # Reshape back to a 2D matrix
    Y = np.reshape(permuted_X, (np.prod(dim), np.prod(dim)))
    
    return Y


# Background: In quantum mechanics, the partial trace is an operation used to trace out (or discard) certain subsystems of a composite quantum state, 
# resulting in a reduced density matrix that describes the remaining subsystems. This is useful when we are interested in the state of a subsystem 
# without considering the rest of the system. Mathematically, if a composite system is described by a density matrix, the partial trace over a 
# subsystem is obtained by summing over the degrees of freedom of that subsystem. For a state with multiple subsystems, the partial trace can be 
# computed by reshaping the density matrix into a higher-dimensional array, permuting the axes to bring the traced-out subsystems together, 
# and then summing over those axes.

def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''


    # Check if the matrix is empty
    if X.size == 0:
        return np.array([[]])

    # Check if the matrix is square and dimensions match the product of subsystem dimensions
    if X.shape[0] != X.shape[1] or X.shape[0] != np.prod(dim):
        raise ValueError("Input matrix must be square and match the product of the subsystem dimensions.")

    # Check for valid subsystem indices
    if any(s >= len(dim) or s < 0 for s in sys):
        raise IndexError("Subsystem index out of range.")

    # Calculate the number of subsystems
    num_subsystems = len(dim)
    
    # Determine the subsystems to keep
    keep = [i for i in range(num_subsystems) if i not in sys]
    
    # Permute the subsystems so that the ones to be traced out are at the end
    perm = keep + sys
    
    # Reshape X according to the dimensions of the subsystems
    reshaped_X = X.reshape([dim[i] for i in range(num_subsystems)] * 2)
    
    # Permute the axes to align for tracing out
    axes = np.concatenate([np.array(keep) * 2, np.array(sys) * 2 + np.array(keep)])
    permuted_X = reshaped_X.transpose(axes)
    
    # Calculate the dimensions of the subsystems to keep and trace out
    dim_keep = [dim[i] for i in keep]
    dim_trace = [dim[i] for i in sys]
    
    # Trace out the subsystems
    traced_shape = dim_keep + dim_keep
    result = permuted_X.reshape(dim_keep + dim_trace + dim_keep + dim_trace)
    for i in range(len(sys)):
        result = np.trace(result, axis1=len(dim_keep) + i, axis2=len(dim_keep) + len(dim_trace) + i)
    
    # Reshape back to a 2D matrix
    result_dim = np.prod(dim_keep)
    Y = result.reshape((result_dim, result_dim))
    
    return Y


# Background: In quantum mechanics, the von Neumann entropy is a measure of the uncertainty or disorder of a quantum state, 
# analogous to the classical Shannon entropy. It is defined for a density matrix ρ as S(ρ) = -Tr(ρ log(ρ)), where Tr denotes 
# the trace operation and log is the matrix logarithm. The von Neumann entropy quantifies the amount of quantum information 
# or entanglement in a quantum state. For a pure state, the entropy is zero, indicating no uncertainty, while for a mixed 
# state, the entropy is positive, reflecting the degree of mixedness or uncertainty in the state.



def entropy(rho):
    '''Inputs:
    rho: 2d array of floats with equal dimensions, the density matrix of the state
    Output:
    en: quantum (von Neumann) entropy of the state rho, float
    '''
    # Ensure rho is a valid density matrix
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("Density matrix must be square.")
    
    # Check if the matrix is Hermitian (symmetric in the real case)
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("Density matrix must be Hermitian.")
    
    # Compute the eigenvalues of the density matrix
    eigenvalues = np.linalg.eigvalsh(rho)
    
    # Check for negative eigenvalues which are not allowed in a valid density matrix
    if np.any(eigenvalues < 0):
        raise ValueError("Density matrix cannot have negative eigenvalues.")
    
    # Filter out zero eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    # If all eigenvalues are zero, raise an error
    if eigenvalues.size == 0:
        raise ValueError("Density matrix cannot have all zero eigenvalues.")
    
    # Calculate the von Neumann entropy
    en = -np.sum(eigenvalues * np.log(eigenvalues))
    
    return en


# Background: In quantum mechanics, the generalized amplitude damping channel (GADC) is a model that describes the interaction of a quantum system with a thermal environment at a finite temperature. This channel is characterized by two parameters: the damping parameter γ, which represents the probability of energy dissipation, and the thermal parameter N, which represents the average number of thermal excitations in the environment. The GADC is described by four Kraus operators, which are 2x2 matrices that act on a single qubit. These operators are used to model the effect of the channel on a quantum state. The Kraus operators for the GADC are given by:
# K1 = sqrt(1-N) * (|0⟩⟨0| + sqrt(1-γ) * |1⟩⟨1|)
# K2 = sqrt(γ(1-N)) * |0⟩⟨1|
# K3 = sqrt(N) * (sqrt(1-γ) * |0⟩⟨0| + |1⟩⟨1|)
# K4 = sqrt(γN) * |1⟩⟨0|
# These operators satisfy the completeness relation ∑_i K_i^† K_i = I, where I is the identity operator.

def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''


    if not isinstance(gamma, (int, float)):
        raise TypeError("Gamma must be a number.")
    if not isinstance(N, (int, float)):
        raise TypeError("N must be a number.")

    if gamma < 0 or gamma > 1:
        raise ValueError("Gamma must be between 0 and 1.")
    if N < 0 or N > 1:
        raise ValueError("N must be between 0 and 1.")

    # Define the basis states |0⟩ and |1⟩
    ket_0 = np.array([[1], [0]], dtype=complex)
    ket_1 = np.array([[0], [1]], dtype=complex)

    # Define the projectors |0⟩⟨0| and |1⟩⟨1|
    proj_0 = ket_0 @ ket_0.conj().T
    proj_1 = ket_1 @ ket_1.conj().T

    # Calculate the Kraus operators
    K1 = np.sqrt(1 - N) * (proj_0 + np.sqrt(1 - gamma) * proj_1)
    K2 = np.sqrt(gamma * (1 - N)) * (ket_0 @ ket_1.conj().T)
    K3 = np.sqrt(N) * (np.sqrt(1 - gamma) * proj_0 + proj_1)
    K4 = np.sqrt(gamma * N) * (ket_1 @ ket_0.conj().T)

    # Return the list of Kraus operators
    kraus = [K1, K2, K3, K4]
    return kraus



# Background: In quantum information theory, the reverse coherent information of a bipartite quantum state ρ is a measure of the quantum correlations between two subsystems A and B. It is defined as the difference between the von Neumann entropy of subsystem A and the von Neumann entropy of the entire system AB. The von Neumann entropy, S(ρ), is a measure of the uncertainty or disorder of a quantum state, analogous to the classical Shannon entropy. It is calculated as S(ρ) = -Tr(ρ log(ρ)), where Tr denotes the trace operation and log is the matrix logarithm. The negative of the reverse coherent information is used to quantify the loss of quantum information when one part of a bipartite state is sent through a quantum channel, such as the generalized amplitude damping channel (GADC). The GADC is characterized by damping parameters γ and N, which describe the interaction of a quantum system with a thermal environment.

def neg_rev_coh_info(p, g, N):
    '''Calculates the negative of coherent information of the output state
    Inputs:
    p: float, parameter for the input state
    g: float, damping parameter
    N: float, thermal parameter
    Outputs:
    neg_I_c: float, negative of coherent information of the output state
    '''
    # Define the initial state |ψ⟩ = sqrt(1-p)|00⟩ + sqrt(p)|11⟩
    psi = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(p)]])
    rho_AB = np.kron(psi, psi.conj().T)

    # Get the Kraus operators for the GADC
    kraus_ops = generalized_amplitude_damping_channel(g, N)

    # Apply the GADC to the first qubit of the state
    rho_out = np.zeros_like(rho_AB, dtype=complex)
    for K in kraus_ops:
        K_full = np.kron(K, np.eye(2))  # Apply K to the first qubit
        rho_out += K_full @ rho_AB @ K_full.conj().T

    # Calculate the von Neumann entropy of the output state S(AB)
    S_AB = entropy(rho_out)

    # Calculate the reduced density matrix for subsystem A
    rho_A = partial_trace(rho_out, [1], [2, 2])

    # Calculate the von Neumann entropy of subsystem A, S(A)
    S_A = entropy(rho_A)

    # Calculate the negative of the reverse coherent information
    neg_I_R = S_AB - S_A

    return neg_I_R

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.8', 3)
target = targets[0]

p = 0.477991
g = 0.2
N = 0.4
assert np.allclose(neg_rev_coh_info(p,g,N), target)
target = targets[1]

p = 0.407786
g = 0.2
N = 0.1
assert np.allclose(neg_rev_coh_info(p,g,N), target)
target = targets[2]

p = 0.399685
g = 0.4
N = 0.2
assert np.allclose(neg_rev_coh_info(p,g,N), target)
