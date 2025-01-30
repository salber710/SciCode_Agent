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


    # Calculate the total dimension of the system
    total_dim = rho.shape[0]

    if sys is None and dim is None:
        # If no specific subsystem is given, assume the channel acts on the entire system
        sys = [i for i in range(int(np.log2(total_dim)))]
        dim = [2] * len(sys)

    # Calculate the dimension of the subsystems for the channel
    sys_dim = np.prod([dim[i] for i in sys])

    # Calculate the dimension of the rest of the system
    rest_dim = int(total_dim / sys_dim)

    # Initialize the transformed density matrix
    transformed_rho = np.zeros((total_dim, total_dim), dtype=np.complex128)

    # Function to apply a single Kraus operator
    def channel_on_subsystem(K_i, rho, sys_dim, rest_dim):
        # Reshape the density matrix to separate the subsystem
        reshaped_rho = np.reshape(rho, (sys_dim, rest_dim, sys_dim, rest_dim))
        # Apply the Kraus operator
        intermediate_rho = np.einsum('ab,bjkl->ajkl', K_i, reshaped_rho)
        intermediate_rho = np.einsum('ajkl,cb->ajkc', intermediate_rho, np.conj(K_i))
        return np.reshape(intermediate_rho, (total_dim, total_dim))

    # Iterate over all Kraus operators
    for K_i in K:
        transformed_rho += channel_on_subsystem(K_i, rho, sys_dim, rest_dim)

    return transformed_rho


try:
    targets = process_hdf5_to_tuple('11.4', 3)
    target = targets[0]
    K = [np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]])]
    rho = np.ones((2,2))/2
    assert np.allclose(apply_channel(K, rho, sys=None, dim=None), target)

    target = targets[1]
    K = [np.sqrt(0.8)*np.eye(2),np.sqrt(0.2)*np.array([[0,1],[1,0]])]
    rho = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    assert np.allclose(apply_channel(K, rho, sys=[2], dim=[2,2]), target)

    target = targets[2]
    K = [np.sqrt(0.8)*np.eye(2),np.sqrt(0.2)*np.array([[0,1],[1,0]])]
    rho = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    assert np.allclose(apply_channel(K, rho, sys=[1,2], dim=[2,2]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e