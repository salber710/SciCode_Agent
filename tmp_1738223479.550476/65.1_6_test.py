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

    # Convert all input matrices to numpy arrays
    arrays = [np.asarray(arg) for arg in args]

    # Calculate the total shape of the resulting tensor product
    result_shape = tuple(np.prod(dim) for dim in zip(*[arr.shape for arr in arrays]))

    # Initialize the result as a 1x1 matrix containing a single 1
    M = np.array([[1.0]])

    # Iterate over each matrix and compute the Kronecker product
    for array in arrays:
        # Repeat the dimensions of the current result to match the new shape
        M = np.repeat(M, array.shape[0], axis=0)
        M = np.repeat(M, array.shape[1], axis=1)

        # Multiply by the current array, reshaped for element-wise multiplication
        M *= np.tile(array, (M.shape[0] // array.shape[0], M.shape[1] // array.shape[1]))

    # Return the final result reshaped to the calculated shape
    return M.reshape(result_shape)


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