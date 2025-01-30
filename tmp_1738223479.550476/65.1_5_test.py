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
    
    # Convert all inputs to numpy arrays if they aren't already
    matrices = [np.asarray(arg) for arg in args]

    # Recursive function to compute the tensor product
    def recursive_tensor_product(matrices):
        if len(matrices) == 1:
            return matrices[0]
        else:
            A = matrices[0]
            B = recursive_tensor_product(matrices[1:])
            m, n = A.shape
            p, q = B.shape
            # Perform element-wise multiplication and reshape
            return np.tensordot(A, B, axes=0).reshape(m * p, n * q)
    
    return recursive_tensor_product(matrices)


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