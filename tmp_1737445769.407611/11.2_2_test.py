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



# Background: In quantum mechanics, a maximally entangled state is a state of a composite quantum system with maximal entanglement. 
# A common example is the Bell state, which is a two-qubit entangled state. In a bipartite maximally entangled state, the two subsystems 
# have their states perfectly correlated in a way that cannot be described independently. The multi-rail encoding is a method of encoding 
# quantum information where the logical qubit is distributed across multiple physical systems or "rails". With m-rail encoding, each party 
# in a bipartite system is represented with m rails. The task here is to construct the density matrix of a bipartite maximally entangled 
# state where each subsystem is encoded using m rails. This involves creating a superposition of states where the logical state is equally 
# distributed over the available rails.

def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''




    # Number of dimensions for the encoding
    dim = 2**rails

    # Create the maximally entangled state |Φ+⟩ = (1/sqrt(dim)) * sum |ii⟩
    entangled_state = np.zeros((dim, dim))
    for i in range(dim):
        entangled_state[i, i] = 1

    # Normalize the entangled state
    entangled_state /= np.sqrt(dim)

    # Create the density matrix: ρ = |Φ+⟩⟨Φ+|
    density_matrix = np.outer(entangled_state.flatten(), entangled_state.flatten().conj())

    return density_matrix


try:
    targets = process_hdf5_to_tuple('11.2', 3)
    target = targets[0]
    assert np.allclose(multi_rail_encoding_state(1), target)

    target = targets[1]
    assert np.allclose(multi_rail_encoding_state(2), target)

    target = targets[2]
    assert np.allclose(multi_rail_encoding_state(3), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e