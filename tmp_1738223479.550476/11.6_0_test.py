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



# Background: In quantum information theory, generalized amplitude damping channels are used to model the interaction of a quantum system with a thermal environment. These channels are parameterized by a damping parameter, gamma, and a thermal parameter, N, which describe the probability of energy dissipation and the degree of thermal noise, respectively. The Kraus operators for these channels describe how the quantum state evolves under the channel's influence. In this problem, a multi-rail encoded quantum state is sent through multiple generalized amplitude damping channels. Each rail is independently affected by these channels, and the task is to compute the final state after passing through these channels. The function `output_state` applies these channels to the multi-rail encoded state to generate the output state.

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



    def generalized_amplitude_damping_channel(gamma, N):
        def create_2x2_matrix(a, b, c, d):
            return np.array([[a, b], [c, d]])
        
        a1 = np.sqrt(1 - N)
        a2 = np.sqrt(1 - N) * np.sqrt(gamma)
        a3 = np.sqrt(N) * np.sqrt(1 - gamma)
        a4 = np.sqrt(N) * np.sqrt(gamma)
        
        A1 = create_2x2_matrix(a1, 0, 0, a1 * np.sqrt(1 - gamma))
        A2 = create_2x2_matrix(0, a2, 0, 0)
        A3 = create_2x2_matrix(a3, 0, 0, np.sqrt(N))
        A4 = create_2x2_matrix(0, 0, a4, 0)
        
        return [A1, A2, A3, A4]

    def multi_rail_encoding_state(rails):
        dim_party = 2 ** rails
        entangled_state = np.zeros(dim_party * dim_party, dtype=np.complex128)
        for i in range(dim_party):
            entangled_state[i * dim_party + i] = 1.0
        entangled_state /= np.sqrt(dim_party)
        density_matrix = np.outer(entangled_state, np.conj(entangled_state))
        return density_matrix

    def apply_channel(K, rho, sys=None, dim=None):
        total_dim = rho.shape[0]
        if sys is None and dim is None:
            sys = [int(np.log2(total_dim))]
            dim = [total_dim]
        
        sys_dim = np.prod([dim[i] for i in sys])
        rest_dim = total_dim // sys_dim
        final_rho = np.zeros((total_dim, total_dim), dtype=np.complex128)

        def apply_single_kraus_op(K_i, rho, sys_dim, rest_dim):
            reshaped_rho = np.reshape(rho, (rest_dim, sys_dim, rest_dim, sys_dim))
            temp_rho = np.einsum('ab,bjkl->ajkl', K_i, reshaped_rho)
            temp_rho = np.einsum('ajkl,cb->ajkc', temp_rho, np.conj(K_i))
            return np.reshape(temp_rho, (total_dim, total_dim))

        for K_i in K:
            final_rho += apply_single_kraus_op(K_i, rho, sys_dim, rest_dim)

        return final_rho

    # Step 1: Create the initial multi-rail entangled state
    initial_state = multi_rail_encoding_state(rails)

    # Step 2: Obtain the Kraus operators for both sets of channels
    kraus_1 = generalized_amplitude_damping_channel(gamma_1, N_1)
    kraus_2 = generalized_amplitude_damping_channel(gamma_2, N_2)

    # Step 3: Apply the first set of channels to the first m rails
    state_after_first = initial_state
    dim = [2] * (2 * rails)
    for i in range(rails):
        state_after_first = apply_channel(kraus_1, state_after_first, [i], dim)

    # Step 4: Apply the second set of channels to the second m rails
    final_state = state_after_first
    for i in range(rails, 2 * rails):
        final_state = apply_channel(kraus_2, final_state, [i], dim)

    return np.real(final_state)


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