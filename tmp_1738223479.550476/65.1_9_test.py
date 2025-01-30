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


try:
    targets = process_hdf5_to_tuple('65.1', 3)
    target = targets[0]
    assert np.allclose(tensor([0,1],[0,1]), target)

    target = targets[1]
    assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)

    target = targets[2]
    assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e