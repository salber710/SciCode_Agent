from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg



# Background: In quantum mechanics, the concept of a ket vector |j⟩ is fundamental to the representation of quantum states in a Hilbert space.
# A basis vector |j⟩ in a d-dimensional space is a vector where all elements are zero except for the j-th element, which is one.
# The tensor product of vectors is a way to construct a new vector space that combines multiple vector spaces, representing the joint state of multiple systems.
# If d is an integer and j is a list [j1, j2, ..., jn], the task is to compute the tensor product of vectors in a d-dimensional space.
# If d is a list [d1, d2, ..., dn], the task is to compute the tensor product where each vector |ji⟩ is a basis vector in its respective di-dimensional space.




def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int) and isinstance(args, list):
        # Case where d is an int and j is a list
        # Create a standard basis vector for each j_i in the list
        basis_vectors = [np.eye(dim)[:, j_i] for j_i in args]
    elif isinstance(dim, list) and isinstance(args, list) and len(dim) == len(args):
        # Case where d is a list and j is a list of the same length
        # Create a basis vector for each dimension d_i and index j_i
        basis_vectors = [np.eye(dim_i)[:, j_i] for dim_i, j_i in zip(dim, args)]
    else:
        raise ValueError("Invalid input: dim and args must be compatible.")
    
    # Compute the tensor product of all the basis vectors
    out = basis_vectors[0]
    for vector in basis_vectors[1:]:
        out = np.kron(out, vector)
    
    return out


try:
    targets = process_hdf5_to_tuple('11.1', 3)
    target = targets[0]
    assert np.allclose(ket(2, 0), target)

    target = targets[1]
    assert np.allclose(ket(2, [1,1]), target)

    target = targets[2]
    assert np.allclose(ket([2,3], [0,1]), target)

