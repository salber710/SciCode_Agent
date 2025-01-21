try:
    import numpy as np
    import itertools
    import scipy.linalg
    
    
    
    # Background: In quantum mechanics, a ket, denoted as |j⟩, is a vector in a complex vector space that represents the state of a quantum system. 
    # When dealing with quantum systems of multiple particles or subsystems, the state is often represented as a tensor product of the states of 
    # individual subsystems. In such scenarios, a standard basis vector in a d-dimensional space is a vector that has a 1 in the j-th position 
    # and 0 elsewhere. If j is a list, the goal is to construct a larger vector space that is the tensor product of individual spaces specified 
    # by dimensions in d. The tensor product combines the spaces into one larger space, and the corresponding ket represents a state in this 
    # composite space.
    
    def ket(dim, args):
        '''Input:
        dim: int or list, dimension of the ket
        args: int or list, the i-th basis vector
        Output:
        out: dim dimensional array of float, the matrix representation of the ket
        '''
    
    
        # Helper function to create a standard basis vector in a given dimension
        def standard_basis_vector(d, j):
            v = np.zeros(d)
            v[j] = 1
            return v
        
        if isinstance(dim, int) and isinstance(args, int):
            # Single dimension and single index
            return standard_basis_vector(dim, args)
        
        elif isinstance(dim, list) and isinstance(args, list):
            # Multiple dimensions and multiple indices
            if len(dim) != len(args):
                raise ValueError("Length of dimensions and indices must match")
            
            # Calculate the tensor product of basis vectors
            vectors = [standard_basis_vector(d, j) for d, j in zip(dim, args)]
            result = vectors[0]
            for vector in vectors[1:]:
                result = np.kron(result, vector)
            return result
        
        else:
            raise TypeError("Both dim and args should be either int or list")
        
        return out
    

    from scicode.parse.parse import process_hdf5_to_tuple
    targets = process_hdf5_to_tuple('11.1', 3)
    target = targets[0]

    assert np.allclose(ket(2, 0), target)
    target = targets[1]

    assert np.allclose(ket(2, [1,1]), target)
    target = targets[2]

    assert np.allclose(ket([2,3], [0,1]), target)
