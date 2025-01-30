from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg

def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    
    def construct_sparse_basis_vector(size, index):
        """Constructs a basis vector using a sparse representation with a dictionary."""
        return {index: 1.0}

    def sparse_tensor_product(vectors):
        """Calculates the tensor product of basis vectors using sparse vector representation."""
        if not vectors:
            return {0: 1.0}
        
        result = vectors[0]
        for vec in vectors[1:]:
            new_result = {}
            for i, val1 in result.items():
                for j, val2 in vec.items():
                    new_result[i * len(vec) + j] = val1 * val2
            result = new_result
        return result

    def to_dense(sparse_vec, total_size):
        """Converts a sparse vector representation to a dense list."""
        dense_vector = [0.0] * total_size
        for index, value in sparse_vec.items():
            dense_vector[index] = value
        return dense_vector

    if isinstance(dim, int):
        # Single dimension case
        sparse_vector = construct_sparse_basis_vector(dim, args)
        return to_dense(sparse_vector, dim)
    elif isinstance(dim, list):
        # Multiple dimensions case
        sparse_vectors = [construct_sparse_basis_vector(d, j) for d, j in zip(dim, args)]
        sparse_result = sparse_tensor_product(sparse_vectors)
        total_size = 1
        for d in dim:
            total_size *= d
        return to_dense(sparse_result, total_size)



def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    # Dimension of the Hilbert space for each party
    dim_party = 2 ** rails

    # Generate the entangled state |ψ⟩ = 1/sqrt(dim_party) * Σ |i⟩|i⟩
    # Using a different approach with combinatorial representation
    entangled_state = np.zeros(dim_party * dim_party, dtype=np.complex128)

    # Use a single loop with direct index calculation for |i⟩|i⟩
    for i in range(dim_party):
        # Map the index i to the corresponding position in the flattened vector
        entangled_state[i * dim_party + i] = 1.0

    # Normalize the entangled state
    entangled_state /= np.sqrt(dim_party)

    # Compute the density matrix ρ = |ψ⟩⟨ψ|
    density_matrix = np.outer(entangled_state, np.conj(entangled_state))

    return density_matrix


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.

    Input:
    args: any number of nd arrays of floats, corresponding to input matrices

    Output:
    M: the tensor product (Kronecker product) of input matrices, 2d array of floats
    '''


    def kronecker_product_direct(A, B):
        """Compute the Kronecker product using direct slicing and multiplication."""
        m, n = A.shape
        p, q = B.shape
        result = np.zeros((m * p, n * q), dtype=A.dtype)
        for i in range(m):
            for j in range(n):
                result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B[:]
        return result

    # Start with a 1x1 identity matrix
    result = np.array([[1.0]], dtype=float)
    for mat in args:
        result = kronecker_product_direct(result, mat)
    
    return result


def apply_channel(K, rho, sys=None, dim=None):


    # Determine the total dimension of the system
    total_dim = rho.shape[0]

    if sys is None and dim is None:
        # If no subsystems specified, apply channel to the entire system
        sys = [int(np.log2(total_dim))]
        dim = [total_dim]

    # Calculate dimensions of the subsystems to apply the channel
    sys_dim = np.prod([dim[i] for i in sys])
    rest_dim = total_dim // sys_dim

    # Initialize the output density matrix
    final_rho = np.zeros((total_dim, total_dim), dtype=np.complex128)

    # Helper function to apply a single Kraus operator
    def apply_single_kraus_op(K_i, rho, sys_dim, rest_dim):
        # Reshape the density matrix to separate the subsystem
        reshaped_rho = np.reshape(rho, (rest_dim, sys_dim, rest_dim, sys_dim))
        
        # Apply the Kraus operator using einsum for clarity
        temp_rho = np.einsum('ab,bjkl->ajkl', K_i, reshaped_rho)
        temp_rho = np.einsum('ajkl,cb->ajkc', temp_rho, np.conj(K_i))
        
        # Reshape back to the original dimensions
        return np.reshape(temp_rho, (total_dim, total_dim))

    # Apply each Kraus operator and accumulate the result
    for K_i in K:
        final_rho += apply_single_kraus_op(K_i, rho, sys_dim, rest_dim)

    return final_rho


def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 lists of floats, [A1, A2, A3, A4]
    '''
    
    # Use a different method involving a nested function for creating operators
    def create_2x2_matrix(a, b, c, d):
        """Helper function to generate a 2x2 matrix."""
        return [[a, b], [c, d]]
    
    # Directly calculate the components of the Kraus operators
    a1 = (1 - N)**0.5
    a2 = (1 - N)**0.5 * gamma**0.5
    a3 = N**0.5 * (1 - gamma)**0.5
    a4 = N**0.5 * gamma**0.5
    
    # Construct the Kraus operators using the helper function
    A1 = create_2x2_matrix(a1, 0, 0, a1 * (1 - gamma)**0.5)
    A2 = create_2x2_matrix(0, a2, 0, 0)
    A3 = create_2x2_matrix(a3, 0, 0, N**0.5)
    A4 = create_2x2_matrix(0, 0, a4, 0)
    
    # Return the list of Kraus operators
    return [A1, A2, A3, A4]


def output_state(rails, gamma_1, N_1, gamma_2, N_2):


    def kraus_operators(gamma, N):
        sqrt_gamma = np.sqrt(gamma)
        sqrt_1_minus_gamma = np.sqrt(1 - gamma)
        sqrt_N = np.sqrt(N)
        sqrt_1_minus_N = np.sqrt(1 - N)

        K1 = np.array([[sqrt_1_minus_N, 0], [0, sqrt_1_minus_gamma]], dtype=np.complex128)
        K2 = np.array([[0, sqrt_gamma * sqrt_N], [0, 0]], dtype=np.complex128)
        K3 = np.array([[0, 0], [sqrt_1_minus_gamma * sqrt_N, 0]], dtype=np.complex128)
        K4 = np.array([[sqrt_N, 0], [0, sqrt_gamma * sqrt_1_minus_N]], dtype=np.complex128)

        return [K1, K2, K3, K4]

    def initial_superposition_state(rails):
        size = 2 ** (2 * rails)
        state_vector = np.ones(size, dtype=np.complex128) / np.sqrt(size)
        return np.outer(state_vector, state_vector.conj())

    def apply_kraus_set(state, kraus_ops, rail_idx, total_rails):
        dim = 2 ** (2 * total_rails)
        new_state = np.zeros((dim, dim), dtype=np.complex128)

        for K in kraus_ops:
            full_K = np.eye(dim, dtype=np.complex128)
            start, end = 2 * rail_idx, 2 * (rail_idx + 1)
            full_K[start:end, start:end] = K
            new_state += full_K @ state @ full_K.conj().T

        return new_state

    # Step 1: Initialize the state in a superposition
    state = initial_superposition_state(rails)

    # Step 2: Retrieve the Kraus operators for the two sets of channels
    kraus_ops_1 = kraus_operators(gamma_1, N_1)
    kraus_ops_2 = kraus_operators(gamma_2, N_2)

    # Step 3: Apply the first set of channels to the first half of the rails
    for rail in range(rails):
        state = apply_kraus_set(state, kraus_ops_1, rail, 2 * rails)

    # Step 4: Apply the second set of channels to the second half of the rails
    for rail in range(rails):
        state = apply_kraus_set(state, kraus_ops_2, rail + rails, 2 * rails)

    return np.real(state)



def measurement(rails):
    '''Returns the measurement projector
    Input:
    rails: int, number of rails
    Output:
    global_proj: ( 2**(2*rails), 2**(2*rails) ) dimensional array of floats
    '''

    # Total number of qubits in the system
    total_qubits = 2 * rails

    # Initialize the global projector as a zero matrix
    global_proj = np.zeros((2**total_qubits, 2**total_qubits))

    # Generate all valid indices using list comprehension and bitwise operations
    valid_indices = [(1 << i) | (1 << (rails + j)) for i in range(rails) for j in range(rails)]
    
    # Iterate over the valid indices and set the corresponding elements in the projector matrix
    for index in valid_indices:
        global_proj[index, index] = 1.0

    return global_proj


def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    perm: list of int containing the desired order
    dim: list of int containing the dimensions of all subsystems.
    Output:
    Y: 2d array of floats with equal dimensions, the density matrix of the permuted state
    '''



    # Verify the input dimensions
    total_dim = reduce(lambda x, y: x * y, dim)
    assert X.shape == (total_dim, total_dim), "Input matrix X must be square with side length equal to the product of dimensions."

    # Compute the cumulative product of dimensions for index conversion
    cumul_dim = [1] + list(np.cumprod(dim[:-1]))

    # Function to map a flat index to its permuted form
    def permute_index(index, cumul_dim, perm, dim):
        multi_index = [(index // cumul_dim[i]) % dim[i] for i in range(len(dim))]
        permuted_multi_index = [multi_index[perm[i]] for i in range(len(perm))]
        new_index = sum(permuted_multi_index[i] * cumul_dim[i] for i in range(len(perm)))
        return new_index

    # Initialize the output matrix
    Y = np.zeros_like(X)

    # Iterate through each element of the matrix and map to new indices
    for i in range(total_dim):
        for j in range(total_dim):
            new_i = permute_index(i, cumul_dim, perm, dim)
            new_j = permute_index(j, cumul_dim, perm, dim)
            Y[new_i, new_j] = X[i, j]

    return Y


def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''


    # Total number of subsystems
    num_subsystems = len(dim)
    
    # Calculate the dimensions of the subsystems to trace out and keep
    dim_trace = np.prod([dim[i] for i in sys])
    dim_keep = np.prod([dim[i] for i in range(num_subsystems) if i not in sys])
    
    # Generate permutation order to move traced subsystems to the start
    perm_order = sys + [i for i in range(num_subsystems) if i not in sys]
    
    # Use syspermute to reorder the subsystems
    permuted_X = syspermute(X, perm_order, dim)
    
    # Reshape to separate traced and kept systems
    reshaped_X = np.reshape(permuted_X, (dim_trace, dim_keep, dim_trace, dim_keep))
    
    # Perform the partial trace by using a sum along the appropriate axes
    identity_trace = np.eye(dim_trace)
    traced_X = np.einsum('aibj,ai->bj', reshaped_X, identity_trace)
    
    return traced_X



def entropy(rho):
    '''Inputs:
    rho: 2d array of floats with equal dimensions, the density matrix of the state
    Output:
    en: quantum (von Neumann) entropy of the state rho, float
    '''



    # Calculate the eigenvalues of the density matrix using 'eigvalsh'
    eigenvalues = scipy.linalg.eigvalsh(rho)

    # Define a function to calculate entropy contribution from a single eigenvalue
    def entropy_contribution(lmbda):
        return 0 if lmbda <= 0 else -lmbda * np.log2(lmbda)

    # Use map to apply the entropy contribution function to all eigenvalues
    contributions = map(entropy_contribution, eigenvalues)

    # Sum up the contributions to get the total entropy
    en = sum(contributions)

    return en


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e