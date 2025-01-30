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
    # Use the Schur decomposition to obtain eigenvalues
    T, Z = np.linalg.schur(rho)
    eigenvalues = np.diag(T)
    
    # Filter out negligible imaginary parts and zero eigenvalues
    eigenvalues = np.real(eigenvalues[np.isclose(np.imag(eigenvalues), 0)])
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    # Compute the von Neumann entropy
    en = -np.sum(eigenvalues * np.log(eigenvalues))
    
    return en


try:
    targets = process_hdf5_to_tuple('71.6', 3)
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