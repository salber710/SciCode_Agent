from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg



def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''




    # Check if dim is a list
    if isinstance(dim, list):
        # If both dim and args are lists, they should be of the same length
        if len(dim) != len(args):
            raise ValueError("Length of dim and args lists must be the same")
        
        # Initialize the result as a tensor product of individual basis vectors
        out = np.array([1.0])  # Start with a scalar value for tensor product
        for d, j in zip(dim, args):
            basis_vector = np.zeros(d)
            basis_vector[j] = 1.0
            out = np.kron(out, basis_vector)  # Compute the Kronecker product
        
    else:  # dim is assumed to be an int
        # Initialize a basis vector with zeros
        out = np.zeros(dim)
        # Set the j-th position to 1
        out[args] = 1.0

    return out


try:
    targets = process_hdf5_to_tuple('11.1', 3)
    target = targets[0]
    assert np.allclose(ket(2, 0), target)

    target = targets[1]
    assert np.allclose(ket(2, [1,1]), target)

    target = targets[2]
    assert np.allclose(ket([2,3], [0,1]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e