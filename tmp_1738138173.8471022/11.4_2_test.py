from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg

def ket(dim, args):


    def create_basis_vector(d, i):
        # Create a d-dimensional zero vector and set the i-th position to 1
        vec = np.zeros(d)
        vec[i] = 1
        return vec

    if isinstance(args, list):
        # Compute the tensor product of basis vectors for each dimension and index in args
        result = create_basis_vector(dim[0], args[0])
        for d, i in zip(dim[1:], args[1:]):
            result = np.kron(result, create_basis_vector(d, i))
    else:
        # Single dimension and index
        result = create_basis_vector(dim, args)

    return result



def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    # Dimension of each subsystem
    dim = 2 ** rails
    
    # Create a maximally entangled state using a Hadamard-like approach
    entangled_state = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            entangled_state[i, j] = 1 if (i ^ j) == 0 else 0
    
    # Flatten the matrix to create the state vector
    entangled_state = entangled_state.flatten()
    
    # Normalize the state vector
    entangled_state /= np.sqrt(dim)
    
    # Create the density matrix by taking the outer product of the state vector with itself
    state = np.outer(entangled_state, entangled_state)
    
    return state


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''


    # Compute the tensor product using a loop, but process in pairs from end to start
    if not args:
        return np.array([[]])  # Return an empty array if no arguments are provided

    # Start with the last matrix/vector
    M = args[-1]

    # Process in pairs
    for i in range(len(args) - 2, -1, -2):
        if i - 1 >= 0:
            M = np.kron(np.kron(args[i-1], args[i]), M)
        else:
            M = np.kron(args[i], M)

    return M



def apply_channel(K, rho, sys=None, dim=None):


    def tensor_product(*matrices):
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result

    if sys is None or dim is None:
        # Apply channel to the entire system
        return sum(np.dot(k, np.dot(rho, k.conj().T)) for k in K)

    # Initialize the resulting state
    result_rho = np.zeros_like(rho)
    identity_matrices = [np.eye(d) for d in dim]

    # Apply channel to specified subsystems
    for k in K:
        for index_combination in np.ndindex(*dim):
            modified_matrices = identity_matrices.copy()
            apply_kraus = True
            for i, idx in enumerate(index_combination):
                if i in sys:
                    if idx == 0:  # Apply Kraus operator only at the first index of each subsystem
                        modified_matrices[i] = k
                    else:
                        apply_kraus = False
                        break
            if apply_kraus:
                kraus_op = tensor_product(*modified_matrices)
                result_rho += kraus_op @ rho @ kraus_op.conj().T

    return result_rho


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