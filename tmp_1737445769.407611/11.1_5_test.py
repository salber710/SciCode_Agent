from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg



# Background: In quantum mechanics, a ket vector such as |j⟩ is a column vector that represents a state in a Hilbert space. 
# A standard basis vector in a d-dimensional space is a vector that has a 1 in the j-th position and 0 elsewhere, 
# where j is an index that ranges from 0 to d-1. When dealing with multiple subsystems, each possibly of different dimensions, 
# we use the tensor product to combine these states into a single state that represents the composite system. 
# This involves creating basis vectors in each subsystem's space and then taking their tensor product to form a state in the larger composite space.




def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    
    if isinstance(dim, int):
        # If dim is an int, create a single basis vector in dim-dimensional space
        out = np.zeros((dim,), dtype=float)
        out[args] = 1.0
    else:
        # If dim is a list, create tensor product of basis vectors
        if isinstance(args, int):
            args = [args]  # Make args a list if it's not already

        # Initialize the ket as a scalar 1 (since we'll use np.kron iteratively)
        out = np.array([1.0])

        # Iterate over each dimension and corresponding index
        for d, j in zip(dim, args):
            # Create the basis vector for the current subsystem
            basis_vector = np.zeros((d,), dtype=float)
            basis_vector[j] = 1.0

            # Take the tensor product with the current state
            out = np.kron(out, basis_vector)
        
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