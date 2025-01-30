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
    """
    Applies the channel with Kraus operators in K to the state rho on
    systems specified by the list sys. The dimensions of the subsystems of
    rho are given by dim.
    Inputs:
    K: list of 2d array of floats, list of Kraus operators
    rho: 2d array of floats, input density matrix
    sys: list of int or None, list of subsystems to apply the channel, None means full system
    dim: list of int or None, list of dimensions of each subsystem, None means full system
    Output:
    matrix: output density matrix of floats
    """
    if sys is None or dim is None:
        # If sys or dim is not provided, apply the channel to the entire system.
        return sum(np.dot(K_i, np.dot(rho, K_i.conj().T)) for K_i in K)

    # Calculate the total dimension of the overall system
    total_dim = int(np.prod(dim))
    output_rho = np.zeros((total_dim, total_dim), dtype=complex)

    # Extract dimensions for the targeted subsystem
    target_dims = [dim[i] for i in sys]
    target_dim = int(np.prod(target_dims))

    # Calculate the dimensions of the remaining subsystems
    non_target_dims = [dim[i] for i in range(len(dim)) if i not in sys]
    non_target_dim = int(np.prod(non_target_dims))

    # Helper function to compute tensor product
    def compute_tensor_product(operators):
        result = operators[0]
        for operator in operators[1:]:
            result = np.kron(result, operator)
        return result

    # Apply the channel using the Kraus operators
    for kraus_op in K:
        reshaped_kraus_op = kraus_op.reshape((target_dim, target_dim))
        
        # Build the full operator by inserting the Kraus operator in the right position
        for i in range(non_target_dim):
            block_identity = np.eye(non_target_dim, dtype=complex)
            full_kraus_op = compute_tensor_product([reshaped_kraus_op if j == i else block_identity for j in range(non_target_dim)])
            
            # Apply the full Kraus operator to the density matrix
            temp_rho = full_kraus_op @ rho @ full_kraus_op.conj().T
            output_rho += temp_rho

    return output_rho


try:
    targets = process_hdf5_to_tuple('71.3', 3)
    target = targets[0]
    K = [np.eye(2)]
    rho = np.array([[0.8,0],[0,0.2]])
    assert np.allclose(apply_channel(K, rho, sys=None, dim=None), target)

    target = targets[1]
    K = [np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]])]
    rho = np.ones((2,2))/2
    assert np.allclose(apply_channel(K, rho, sys=None, dim=None), target)

    target = targets[2]
    K = [np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]])]
    rho = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    assert np.allclose(apply_channel(K, rho, sys=[2], dim=[2,2]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e