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


    def kraus_ops(gamma, N):
        sqrt_gamma = np.sqrt(gamma)
        sqrt_1_minus_gamma = np.sqrt(1 - gamma)
        sqrt_N = np.sqrt(N)
        sqrt_1_minus_N = np.sqrt(1 - N)

        return [
            np.array([[sqrt_1_minus_N, 0], [0, sqrt_1_minus_gamma]], dtype=np.complex128),
            np.array([[0, sqrt_gamma * sqrt_N], [0, 0]], dtype=np.complex128),
            np.array([[0, 0], [sqrt_1_minus_gamma * sqrt_N, 0]], dtype=np.complex128),
            np.array([[sqrt_N, 0], [0, sqrt_gamma * sqrt_1_minus_N]], dtype=np.complex128)
        ]

    def initial_multi_rail_state(rails):
        size = 2 ** (2 * rails)
        initial_vector = np.zeros(size, dtype=np.complex128)
        initial_vector[0] = 1.0
        return np.outer(initial_vector, initial_vector.conj())

    def apply_kraus_set(state, kraus_set, rail_index, total_rails):
        dim = 2 ** (2 * total_rails)
        updated_state = np.zeros((dim, dim), dtype=np.complex128)
        for K in kraus_set:
            expanded_K = np.eye(dim, dtype=np.complex128)
            expanded_K[2 * rail_index:2 * (rail_index + 1), 2 * rail_index:2 * (rail_index + 1)] = K
            updated_state += expanded_K @ state @ expanded_K.conj().T
        return updated_state

    # Step 1: Create the initial multi-rail state
    state = initial_multi_rail_state(rails)

    # Step 2: Calculate Kraus operators for both sets of channels
    kraus_1 = kraus_ops(gamma_1, N_1)
    kraus_2 = kraus_ops(gamma_2, N_2)

    # Step 3: Apply the first set of channels to the first m rails
    for rail in range(rails):
        state = apply_kraus_set(state, kraus_1, rail, rails)

    # Step 4: Apply the second set of channels to the second m rails
    for rail in range(rails):
        state = apply_kraus_set(state, kraus_2, rail + rails, rails)

    return np.real(state)


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