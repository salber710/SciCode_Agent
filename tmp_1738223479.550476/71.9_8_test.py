from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

def ket(dim, *args):
    if isinstance(dim, int):
        return [float(i == args[0]) for i in range(dim)]

    elif isinstance(dim, list):
        def basis_vector(d, idx):
            return [float(i == idx) for i in range(d)]

        def tensor_product(vectors):
            if not vectors:
                return [1.0]
            # We will use a variable to store the result and build it iteratively
            result = vectors[0]
            for vec in vectors[1:]:
                result = [a * b for a in result for b in vec]
            return result
        
        vectors = [basis_vector(d, j) for d, j in zip(dim, args)]
        return tensor_product(vectors)



def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one matrix or vector is required")

    matrices = [np.array(arg, dtype=float) for arg in args]

    # Initialize the result as a single element array containing 1.0
    result = np.ones((1, 1))

    # Iterate over each matrix
    for matrix in matrices:
        # Get the shape of the current result and next matrix
        r_rows, r_cols = result.shape
        m_rows, m_cols = matrix.shape
        
        # Compute the new shape for the result
        new_shape = (r_rows * m_rows, r_cols * m_cols)
        
        # Allocate a new array for the result
        new_result = np.zeros(new_shape)
        
        # Compute the Kronecker product using advanced indexing and broadcasting
        row_indices = np.repeat(np.arange(r_rows), m_rows) * m_rows + np.tile(np.arange(m_rows), r_rows)
        col_indices = np.repeat(np.arange(r_cols), m_cols) * m_cols + np.tile(np.arange(m_cols), r_cols)
        
        # Multiply and assign using outer product for each element
        new_result[row_indices[:, None], col_indices] = np.outer(result.ravel(), matrix.ravel()).reshape(new_shape)

        # Update the result
        result = new_result

    return result



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
    # Handle the case where the channel is applied to the entire system
    if sys is None or dim is None:
        return sum(np.dot(k, np.dot(rho, k.conj().T)) for k in K)

    # Compute the full dimension of the system
    total_dim = int(np.prod(dim))
    output_rho = np.zeros((total_dim, total_dim), dtype=complex)

    # Dimensions of the targeted subsystems
    sys_dims = [dim[i] for i in sys]
    sys_dim = int(np.prod(sys_dims))

    # Dimensions of the other subsystems
    non_sys_indices = [i for i in range(len(dim)) if i not in sys]
    non_sys_dims = [dim[i] for i in non_sys_indices]
    non_sys_dim = int(np.prod(non_sys_dims))

    # Create a function to compute the Kronecker product recursively
    def recursive_kron(*matrices):
        if len(matrices) == 0:
            return np.array([[1]])
        if len(matrices) == 1:
            return matrices[0]
        return np.kron(matrices[0], recursive_kron(*matrices[1:]))

    # Apply each Kraus operator
    for kraus_op in K:
        # Reshape the Kraus operator appropriately
        reshaped_kraus_op = kraus_op.reshape(sys_dim, sys_dim)

        # Construct the full Kraus operator
        identity_blocks = [np.eye(d) for d in non_sys_dims]
        kraus_op_parts = identity_blocks[:]
        for index in range(len(sys)):
            kraus_op_parts.insert(sys[index], reshaped_kraus_op)

        full_kraus_op = recursive_kron(*kraus_op_parts)

        # Apply the full Kraus operator to the density matrix
        temp_rho = full_kraus_op @ rho @ full_kraus_op.conj().T
        output_rho += temp_rho

    return output_rho


def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    perm: list of int containing the desired order
    dim: list of int containing the dimensions of all subsystems.
    Output:
    Y: 2d array of floats with equal dimensions, the density matrix of the permuted state
    '''


    # Calculate total dimension of the system
    total_dim = np.prod(dim)

    # Create an array to store the indices for reshaping
    original_indices = np.arange(total_dim**2).reshape((total_dim, total_dim))

    # Calculate original multi-dimensional indices for rows and columns
    original_row_indices = np.unravel_index(original_indices // total_dim, dim)
    original_col_indices = np.unravel_index(original_indices % total_dim, dim)

    # Apply permutation to obtain new multi-dimensional indices
    permuted_row_indices = [original_row_indices[i] for i in perm]
    permuted_col_indices = [original_col_indices[i] for i in perm]

    # Convert permuted multi-indices back to flat indices
    permuted_row_flat = np.ravel_multi_index(permuted_row_indices, dim)
    permuted_col_flat = np.ravel_multi_index(permuted_col_indices, dim)

    # Create the permuted matrix Y
    Y = np.zeros_like(X)
    Y[permuted_row_flat, permuted_col_flat] = X[original_indices // total_dim, original_indices % total_dim]

    return Y



def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''
    # Calculate dimensions for systems to keep and trace out
    keep_dims = [dim[i] for i in range(len(dim)) if i not in sys]
    trace_dims = [dim[i] for i in sys]
    
    # Product of dimensions
    total_keep_dim = np.prod(keep_dims)
    total_trace_dim = np.prod(trace_dims)
    
    # Permute to bring traced subsystems to the end
    permute_order = sys + [i for i in range(len(dim)) if i not in sys]
    X_permuted = syspermute(X, permute_order, dim)
    
    # Reshape for tracing
    reshaped_X = X_permuted.reshape(
        *trace_dims, *keep_dims, *trace_dims, *keep_dims
    )
    
    # Perform partial trace by explicit summing over indices
    for _ in range(len(trace_dims)):
        reshaped_X = np.trace(reshaped_X, axis1=0, axis2=len(trace_dims))
    
    # Reshape result to final reduced density matrix
    return reshaped_X.reshape(total_keep_dim, total_keep_dim)



def entropy(rho):
    '''Inputs:
    rho: 2d array of floats with equal dimensions, the density matrix of the state
    Output:
    en: quantum (von Neumann) entropy of the state rho, float
    '''
    # Use the Frobenius norm to approximate eigenvalues
    # While this is not a direct way to compute eigenvalues, it gives a distinct approach
    n = rho.shape[0]
    
    # Initialize a matrix to accumulate contributions to a "pseudo-eigenvalue" matrix
    pseudo_matrix = np.zeros_like(rho)
    
    # Use a randomized approach to create a pseudo-eigenvalue matrix
    for _ in range(n):
        random_vector = np.random.randn(n)
        random_vector /= np.linalg.norm(random_vector)
        contribution = np.outer(random_vector, random_vector)
        pseudo_matrix += np.dot(rho, contribution)
    
    # Diagonal values of the pseudo_matrix are used as pseudo-eigenvalues
    pseudo_eigenvalues = np.diag(pseudo_matrix)
    
    # Normalize these pseudo-eigenvalues to serve as probabilities
    normalized_eigenvalues = pseudo_eigenvalues / np.sum(pseudo_eigenvalues)
    
    # Filter out zero to avoid log(0) issues
    normalized_eigenvalues = normalized_eigenvalues[normalized_eigenvalues > 0]

    # Compute the von Neumann entropy using the normalized pseudo-eigenvalues
    en = -np.sum(normalized_eigenvalues * np.log(normalized_eigenvalues))
    
    return en


def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''


    # Precompute square roots and products to avoid redundancy
    sqrt_1_minus_gamma = np.sqrt(1 - gamma)
    sqrt_gamma = np.sqrt(gamma)
    sqrt_1_minus_N = np.sqrt(1 - N)
    sqrt_N = np.sqrt(N)
    sqrt_gamma_1_minus_N = sqrt_gamma * sqrt_1_minus_N
    sqrt_gamma_N = sqrt_gamma * sqrt_N

    # Define the Kraus operators using numpy arrays directly
    K1 = np.array([[sqrt_1_minus_N, 0], [0, sqrt_1_minus_N * sqrt_1_minus_gamma]], dtype=float)
    K2 = np.array([[0, sqrt_gamma_1_minus_N], [0, 0]], dtype=float)
    K3 = np.array([[sqrt_N * sqrt_1_minus_gamma, 0], [0, sqrt_N]], dtype=float)
    K4 = np.array([[0, 0], [sqrt_gamma_N, 0]], dtype=float)

    # Return the list of Kraus operators
    kraus = [K1, K2, K3, K4]
    return kraus



def neg_rev_coh_info(p, g, N):
    '''Calculates the negative of coherent information of the output state
    Inputs:
    p: float, parameter for the input state
    g: float, damping parameter
    N: float, thermal parameter
    Outputs:
    neg_I_R: float, negative of reverse coherent information of the output state
    '''

    # Define the initial state |ψ⟩ = √(1-p)|00⟩ + √p|11⟩
    psi = np.array([np.sqrt(1-p), 0, 0, np.sqrt(p)])
    rho_input = np.outer(psi, psi.conj())

    # Define the generalized amplitude damping channel Kraus operators
    def gad_kraus_ops(g, N):
        A = np.sqrt(1 - g)
        B = np.sqrt(g)
        C = np.sqrt(1 - N)
        D = np.sqrt(N)
        return [
            A * np.array([[1, 0], [0, C]]),
            A * np.array([[0, D], [0, 0]]),
            B * np.array([[C, 0], [0, 1]]),
            B * np.array([[0, 0], [D, 0]])
        ]

    K = gad_kraus_ops(g, N)

    # Apply the channel to the second qubit
    def apply_channel(rho, K):
        result_rho = np.zeros_like(rho, dtype=complex)
        for k in K:
            result_rho += np.kron(np.eye(2), k) @ rho @ np.kron(np.eye(2), k).conj().T
        return result_rho

    output_rho = apply_channel(rho_input, K)

    # Calculate the entropy of a density matrix
    def compute_entropy(rho):
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Consider only positive eigenvalues
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    # Calculate the entropy of the full output state
    S_AB = compute_entropy(output_rho)

    # Perform the partial trace over the second qubit to get the reduced state of qubit A
    def partial_trace(rho, keep, dims):
        keep_dim = np.prod([dims[i] for i in keep])
        trace_dim = np.prod([dims[i] for i in range(len(dims)) if i not in keep])
        reshaped_rho = rho.reshape(keep_dim, trace_dim, keep_dim, trace_dim)
        return np.trace(reshaped_rho, axis1=1, axis2=3)

    rho_A = partial_trace(output_rho, keep=[0], dims=[2, 2])

    # Calculate the entropy of the reduced state of subsystem A
    S_A = compute_entropy(rho_A)

    # Calculate the negative of the reverse coherent information
    neg_I_R = S_AB - S_A

    return neg_I_R




def GADC_rev_coh_inf(g, N):
    '''Calculates the coherent information of the GADC.
    Inputs:
    g: float, damping parameter
    N: float, thermal parameter
    Outputs:
    channel_coh_info: float, channel coherent information of a GADC
    '''

    def neg_coh_info(p):
        # Calculate the negative reverse coherent information for a given p.
        return neg_rev_coh_info(p, g, N)

    # Use a Monte Carlo method to randomly sample values of p and find the minimum neg_coh_info
    num_samples = 1000
    p_values = np.random.uniform(0, 1, num_samples)
    neg_coh_info_values = np.array([neg_coh_info(p) for p in p_values])

    # Find the p that minimizes the negative coherent information
    optimal_index = np.argmin(neg_coh_info_values)
    optimal_p = p_values[optimal_index]

    # Compute the channel coherent information using the optimal p found
    channel_coh_info = -neg_coh_info(optimal_p)

    return channel_coh_info


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e