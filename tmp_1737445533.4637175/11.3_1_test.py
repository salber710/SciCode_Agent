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
    
    
    # Background: In quantum computing, a bipartite maximally entangled state is a special quantum state involving two subsystems
    # that are maximally correlated. A common example is the Bell state. In the context of $m$-rail encoding, each party's system
    # is represented using multiple "rails" or qubits. The multi-rail encoding involves distributing the information across multiple
    # qubits per party to potentially increase robustness against certain types of errors. The density matrix of such a state describes
    # the statistical mixtures of quantum states that the system can be in. For a bipartite maximally entangled state with $m$-rail 
    # encoding, the state is constructed in a larger Hilbert space, where each subsystem is represented by $m$ qubits.
    
    def multi_rail_encoding_state(rails):
        '''Returns the density matrix of the multi-rail encoding state
        Input:
        rails: int, number of rails
        Output:
        state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
        '''
    
    
        
        # Dimension of the system for each party
        dimension = 2 ** rails
        
        # Create the maximally entangled state |Φ⟩ = 1/√d * Σ |i⟩|i⟩
        entangled_state = np.zeros((dimension, dimension), dtype=np.float64)
        
        for i in range(dimension):
            entangled_state[i, i] = 1
        
        # Normalize the state
        entangled_state = entangled_state / np.sqrt(dimension)
        
        # Compute the density matrix ρ = |Φ⟩⟨Φ|
        density_matrix = np.outer(entangled_state.flatten(), entangled_state.flatten().conj())
        
        return density_matrix
    
    
    
    # Background: In linear algebra, the tensor product (also known as the Kronecker product) of two matrices is a matrix 
    # that represents the product of the two linear transformations. The tensor product is used to construct larger matrices
    # from smaller ones and is particularly useful in quantum mechanics for representing composite systems. The tensor product 
    # of matrices A (of size m x n) and B (of size p x q) results in a matrix of size (m*p) x (n*q). This operation is 
    # associative, meaning that the tensor product of matrices can be extended to more than two matrices and the order of 
    # operations does not affect the result. In this function, we aim to compute the tensor product of an arbitrary number of 
    # matrices/vectors, which is a common requirement in quantum mechanics and other fields that deal with multilinear algebra.
    
    def tensor(*args):
        '''Takes the tensor product of an arbitrary number of matrices/vectors.
        Input:
        args: any number of nd arrays of floats, corresponding to input matrices
        Output:
        M: the tensor product (kronecker product) of input matrices, 2d array of floats
        '''
    
    
        # Start with the first matrix/vector
        result = args[0]
        
        # Iterate through each remaining matrix/vector and compute the tensor product
        for matrix in args[1:]:
            result = np.kron(result, matrix)
        
        return result
    

    from scicode.parse.parse import process_hdf5_to_tuple
    targets = process_hdf5_to_tuple('11.3', 3)
    target = targets[0]

    assert np.allclose(tensor([0,1],[0,1]), target)
    target = targets[1]

    assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)
    target = targets[2]

    assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)
