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

    def kronecker_recursive(A, B):
        if A.ndim == 1:
            A = A[:, np.newaxis]
        if B.ndim == 1:
            B = B[:, np.newaxis]
        
        m, n = A.shape
        p, q = B.shape
        result = np.zeros((m * p, n * q))

        for i in range(m):
            for j in range(n):
                result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B

        return result

    def compute_tensor_product(matrices):
        if len(matrices) == 1:
            return matrices[0]
        elif len(matrices) == 2:
            return kronecker_recursive(matrices[0], matrices[1])
        else:
            return kronecker_recursive(matrices[0], compute_tensor_product(matrices[1:]))

    matrices = [np.array(arg, dtype=float) for arg in args]
    return compute_tensor_product(matrices)


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