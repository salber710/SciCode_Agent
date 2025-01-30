from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.linalg import sqrtm
import itertools


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one input is required")
    
    # Convert inputs to numpy arrays and store their shapes
    matrices = [np.asarray(arg) for arg in args]
    shapes = [matrix.shape for matrix in matrices]

    # Calculate the shape of the resulting tensor product
    result_shape = (np.prod([shape[0] for shape in shapes]), np.prod([shape[1] for shape in shapes]))

    # Initialize the result as a 1 element matrix containing 1.0
    M = np.array([[1.0]])

    # Compute the Kronecker product using a unique approach with np.add.outer
    for matrix in matrices:
        # Use add.outer to generate a block matrix for the Kronecker product
        M = np.add.outer(M, matrix).reshape(M.shape[0] * matrix.shape[0], M.shape[1] * matrix.shape[1])
    
    return M



def apply_channel(K, rho, sys=None, dim=None):


    def tensor_up(operator, target_dim, full_dim):
        '''Embeds an operator into the full Hilbert space by applying identity operators to other subsystems.'''
        identity = np.eye(full_dim // target_dim)
        return np.kron(operator, identity)

    if sys is None and dim is None:
        # Apply the channel to the entire system
        new_rho = sum(K_i @ rho @ K_i.conj().T for K_i in K)
        return new_rho

    if sys is None or dim is None:
        raise ValueError("Both sys and dim must be provided if not applying to the entire system")

    # Total dimension of the system
    total_dim = int(np.sqrt(rho.size))
    if total_dim != np.prod(dim):
        raise ValueError("Dimensions do not match the size of rho")

    # Initialize new_rho as a zero matrix with the same shape as rho
    new_rho = np.zeros_like(rho, dtype=np.complex128)

    for K_i in K:
        # Start with identity operators on the full system
        operator = np.eye(total_dim, dtype=np.complex128)

        for subsystem in sys:
            # Determine the dimensions of the subsystems before and after the target
            left_dim = np.prod(dim[:subsystem], dtype=int)
            target_dim = dim[subsystem]
            right_dim = np.prod(dim[subsystem+1:], dtype=int)

            # Create the operator for this subsystem
            operator = np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), K_i), np.eye(right_dim, dtype=np.complex128))

            # Apply the operator to the state
            new_rho += operator @ rho @ operator.conj().T

    return new_rho


try:
    targets = process_hdf5_to_tuple('65.2', 3)
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