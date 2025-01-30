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



# Background: In quantum mechanics, a quantum channel can be represented by a set of Kraus operators, which describe the effect of the channel on a quantum state. The density matrix of a quantum state, denoted as rho, can be transformed by these Kraus operators. The transformation is given by the formula: rho' = sum(K_i * rho * K_i^†), where K_i are the Kraus operators and K_i^† is the conjugate transpose of K_i. When applying the channel to specific subsystems of a composite system, we need to consider the tensor product structure of the state and apply the operators only to the specified subsystems. This involves computing partial traces and using the tensor product to correctly position the operators within the larger Hilbert space.

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



    # If sys and dim are None, apply the channel to the entire system
    if sys is None and dim is None:
        new_rho = np.zeros_like(rho, dtype=np.complex128)
        for K_i in K:
            new_rho += K_i @ rho @ K_i.conj().T
        return new_rho

    # Check that sys and dim are provided and are lists
    if sys is None or dim is None:
        raise ValueError("Both sys and dim must be provided if not applying to the entire system")

    # Calculate the total dimension of rho
    total_dim = int(np.sqrt(rho.size))
    if total_dim != np.prod(dim):
        raise ValueError("Dimensions do not match the size of rho")

    # Create identity operators for subsystems that are not affected
    unaffected_subsystems = set(range(len(dim))) - set(sys)
    identity_operators = [np.eye(dim[i]) for i in unaffected_subsystems]

    # Apply the channel to specified subsystems
    new_rho = np.zeros_like(rho, dtype=np.complex128)
    for K_i in K:
        # Initialize the operator for this step
        operator_list = [np.eye(dim[i]) for i in range(len(dim))]
        for idx, sub in enumerate(sys):
            # Modify the operator only for the specified systems
            operator_list[sub] = K_i

        # Construct the full operator using tensor product
        full_operator = operator_list[0]
        for op in operator_list[1:]:
            full_operator = np.kron(full_operator, op)

        # Apply the operator to rho
        new_rho += full_operator @ rho @ full_operator.conj().T

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