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



# Background: In quantum information, a bipartite maximally entangled state is a state of two systems that is maximally entangled, meaning
# that the state cannot be factored into the product of two states of the subsystems. A common example of a maximally entangled state is
# the Bell state. In the context of multi-rail encoding, each party's state is represented using multiple 'rails' (or basis states), 
# allowing for redundancy and robustness in quantum communication. The goal is to construct a density matrix representing such an 
# entangled state for a given number of rails. The density matrix is a square matrix that represents the state of a quantum system, 
# including mixed states.



def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    # Number of dimensions for a single party
    dim = 2**rails
    
    # Create the maximally entangled state vector |Φ⟩, which is a sum of |ii⟩ for i from 0 to dim-1
    state_vector = np.zeros((dim * dim,), dtype=np.complex128)
    for i in range(dim):
        ket_i = np.zeros((dim,), dtype=np.complex128)
        ket_i[i] = 1
        # Tensor product |i⟩|i⟩
        state_vector += np.kron(ket_i, ket_i)
    
    # Normalize the state vector
    state_vector /= np.sqrt(dim)

    # Create the density matrix ρ = |Φ⟩⟨Φ|
    state = np.outer(state_vector, np.conj(state_vector))
    
    return state

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('11.2', 3)
target = targets[0]

assert np.allclose(multi_rail_encoding_state(1), target)
target = targets[1]

assert np.allclose(multi_rail_encoding_state(2), target)
target = targets[2]

assert np.allclose(multi_rail_encoding_state(3), target)
