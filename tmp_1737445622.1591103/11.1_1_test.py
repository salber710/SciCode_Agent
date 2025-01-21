from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg



# Background: In quantum mechanics, a ket |j⟩ is a vector in a Hilbert space representing the state of a quantum system.
# A standard basis vector |j⟩ in a d-dimensional space is a vector with a 1 at the j-th position (0-indexed) and 0 elsewhere.
# The tensor product of vectors is used to describe the combined state of multiple quantum systems.
# If given a list of indices and dimensions, the tensor product of standard basis vectors is computed by constructing each individual basis vector and then combining them using the tensor product operation.




def ket(dim, *args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single space case
        d = dim
        j = args[0]
        # Create a d-dimensional zero vector
        ket_vector = np.zeros((d,))
        # Set the j-th position to 1
        ket_vector[j] = 1.0
        return ket_vector
    
    elif isinstance(dim, list):
        # Multiple spaces case
        d_list = dim
        j_list = args[0]
        assert len(d_list) == len(j_list), "Dimension list and index list must be of the same length"
        
        # Create the ket vector for each dimension and index pair
        ket_vectors = []
        for d, j in zip(d_list, j_list):
            vec = np.zeros((d,))
            vec[j] = 1.0
            ket_vectors.append(vec)
        
        # Compute the tensor product of all the vectors
        out = ket_vectors[0]
        for vec in ket_vectors[1:]:
            out = np.kron(out, vec)
        
        return out


try:
    targets = process_hdf5_to_tuple('11.1', 3)
    target = targets[0]
    assert np.allclose(ket(2, 0), target)

    target = targets[1]
    assert np.allclose(ket(2, [1,1]), target)

    target = targets[2]
    assert np.allclose(ket([2,3], [0,1]), target)

