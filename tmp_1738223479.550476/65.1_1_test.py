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
        # Compute the shape of the resulting matrix
        m, n = A.shape
        p, q = B.shape
        # Create an empty result matrix of the appropriate shape
        result = np.zeros((m * p, n * q), dtype=A.dtype)
        
        # Compute the Kronecker product by iterating over each element of A
        for i in range(m):
            for j in range(n):
                result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B
        
        return result

    # Use reduce to apply the kronecker_product function across all input matrices
    M = reduce(kronecker_product, args)
    
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