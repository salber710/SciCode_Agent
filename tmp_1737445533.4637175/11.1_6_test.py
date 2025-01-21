try:
    import numpy as np
    import itertools
    import scipy.linalg
    
    
    
    # Background: 
    # In quantum mechanics and linear algebra, a standard basis vector in a d-dimensional space is a vector with a 1 in a single position and 0s elsewhere. 
    # When dealing with multiple quantum systems, the overall state is described by the tensor product of individual states. 
    # If we have a list of indices [j_1, j_2, ..., j_n] and a dimension list [d_1, d_2, ..., d_n], the task is to construct the tensor product of the individual basis vectors.
    # Each basis vector |j_i⟩ in a d_i-dimensional space is a vector with 1 at position j_i and 0 elsewhere.
    # The numpy library can efficiently handle tensor products and array manipulations required to construct these vectors.
    
    def ket(dim, args):
        '''Input:
        dim: int or list, dimension of the ket
        args: int or list, the i-th basis vector
        Output:
        out: dim dimensional array of float, the matrix representation of the ket
        '''
    
    
        if isinstance(dim, int):
            # Single dimension, create a single basis vector
            out = np.zeros(dim, dtype=float)
            out[args] = 1.0
        else:
            # Multiple dimensions, create a tensor product of basis vectors
            basis_vectors = []
            for d, j in zip(dim, args):
                basis_vector = np.zeros(d, dtype=float)
                basis_vector[j] = 1.0
                basis_vectors.append(basis_vector)
            
            # Compute the tensor product
            out = basis_vectors[0]
            for bv in basis_vectors[1:]:
                out = np.kron(out, bv)
                
        return out
    

    from scicode.parse.parse import process_hdf5_to_tuple
    targets = process_hdf5_to_tuple('11.1', 3)
    target = targets[0]

    assert np.allclose(ket(2, 0), target)
    target = targets[1]

    assert np.allclose(ket(2, [1,1]), target)
    target = targets[2]

    assert np.allclose(ket([2,3], [0,1]), target)
