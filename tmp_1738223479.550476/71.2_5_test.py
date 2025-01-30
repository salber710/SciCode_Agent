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

    # Convert input matrices to numpy arrays
    matrices = [np.array(arg, dtype=float) for arg in args]

    # Use a different approach by iteratively expanding the result array
    result_shape = (1, 1)
    result = np.array([[1.0]])

    for matrix in matrices:
        new_shape = (result_shape[0] * matrix.shape[0], result_shape[1] * matrix.shape[1])
        new_result = np.zeros(new_shape)

        for i in range(result_shape[0]):
            for j in range(result_shape[1]):
                new_result[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = result[i, j] * matrix

        result_shape = new_shape
        result = new_result

    return result


try:
    targets = process_hdf5_to_tuple('71.2', 3)
    target = targets[0]
    assert np.allclose(tensor([0,1],[0,1]), target)

    target = targets[1]
    assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)

    target = targets[2]
    assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e