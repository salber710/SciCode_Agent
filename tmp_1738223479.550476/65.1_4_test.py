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

    # Convert all inputs to numpy arrays and get their shapes
    arrays = [np.asarray(arg) for arg in args]
    shapes = [arr.shape for arr in arrays]

    # Calculate the resulting shape
    result_shape = (np.prod([shape[0] for shape in shapes]), np.prod([shape[1] for shape in shapes]))

    # Initialize the result as a 1x1 matrix containing a single 1
    M = np.array([[1.0]])

    for array in arrays:
        # Use np.einsum to compute the tensor product
        M = np.einsum('ij,kl->ikjl', M, array).reshape(result_shape)

        # Update the result shape for the next iteration
        result_shape = (M.shape[0], M.shape[1])

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