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
    
    def kronecker_product(A, B):
        # Use broadcasting to compute the Kronecker product in a vectorized manner
        A_shape, B_shape = A.shape, B.shape
        # Reshape A and B to facilitate broadcasting the multiplication
        A_expanded = A[:, :, np.newaxis, np.newaxis]
        B_expanded = B[np.newaxis, np.newaxis, :, :]
        # Compute the product
        product = A_expanded * B_expanded
        # Reshape to the final Kronecker product shape
        return product.reshape(A_shape[0] * B_shape[0], A_shape[1] * B_shape[1])
    
    # Start with the first matrix
    M = args[0]
    
    # Apply kronecker_product iteratively to all input matrices
    for matrix in args[1:]:
        M = kronecker_product(M, matrix)
    
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