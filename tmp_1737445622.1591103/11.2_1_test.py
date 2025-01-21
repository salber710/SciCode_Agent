from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

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



# Background: In quantum information theory, a bipartite maximally entangled state is a state of two quantum systems such that each subsystem is maximally entangled with the other. A common example of such a state is the Bell state or EPR pair. In the context of quantum computing, multi-rail encoding is a method of encoding quantum information across multiple physical rails (qubits) to enhance robustness to certain types of errors. The density matrix representation of a quantum state is used to describe mixed states, but for a pure state like a bipartite entangled state, the density matrix is simply the outer product of the state vector with itself.

def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''



    # Generate the indices for the maximally entangled state
    indices = np.arange(2**rails)  # Possible states for one part
    # Generate the state vector |phi> = 1/sqrt(N) * sum(|i>|i>) for i in range 2**rails
    dimension = 2**rails
    entangled_state = np.zeros(dimension**2, dtype=np.complex128)

    # Construct the entangled state vector
    for i in indices:
        basis_ket = np.zeros(dimension**2, dtype=np.complex128)
        basis_ket[i * (dimension + 1)] = 1
        entangled_state += basis_ket

    entangled_state /= np.sqrt(dimension)  # Normalize the state

    # Compute the density matrix as the outer product of the state with itself
    density_matrix = np.outer(entangled_state, entangled_state.conj())

    return density_matrix.astype(np.float64)


try:
    targets = process_hdf5_to_tuple('11.2', 3)
    target = targets[0]
    assert np.allclose(multi_rail_encoding_state(1), target)

    target = targets[1]
    assert np.allclose(multi_rail_encoding_state(2), target)

    target = targets[2]
    assert np.allclose(multi_rail_encoding_state(3), target)

