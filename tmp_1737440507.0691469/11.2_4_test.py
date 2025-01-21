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



# Background: In quantum mechanics, a bipartite maximally entangled state is a state of two subsystems that is entangled in such a way that 
# the subsystems cannot be described independently. In the context of multi-rail encoding, each party's state is represented over multiple 
# 'rails' or dimensions. The multi-rail encoding maximally entangled state can be represented as a density matrix. A density matrix is a 
# square matrix that describes the statistical state of a quantum system. For a system of n qubits, the density matrix is of size 2^n x 2^n. 
# In this case, we are constructing a maximally entangled state using `rails` number of dimensions for each party, leading to a state space 
# size of 2**(2 * rails) x 2**(2 * rails). The Bell state, for example, can be generalized to higher dimensions using the concept of maximum 
# entanglement where each dimension is equally likely.

def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''



    dim = 2 ** rails  # Dimension of each party's Hilbert space
    total_dim = dim ** 2  # Total dimension for the bipartite system

    # Create the maximally entangled state |Φ+⟩ = (1/sqrt(d)) * sum |i⟩|i⟩, where i runs over all basis states
    entangled_state = np.zeros((total_dim,), dtype=np.complex128)
    
    normalization_factor = 1 / np.sqrt(dim)
    
    for i in range(dim):
        # Constructing the multi-rail encoding basis state |i⟩|i⟩
        # This corresponds to a Kronecker product of |i⟩ and |i⟩ in their respective spaces
        basis_state = np.zeros((total_dim,), dtype=np.complex128)
        basis_state[i * dim + i] = 1  # Place a 1 at the position corresponding to |i⟩|i⟩
        entangled_state += normalization_factor * basis_state

    # Create the density matrix ρ = |Φ+⟩⟨Φ+|
    state = np.outer(entangled_state, np.conj(entangled_state))
    return state

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('11.2', 3)
target = targets[0]

assert np.allclose(multi_rail_encoding_state(1), target)
target = targets[1]

assert np.allclose(multi_rail_encoding_state(2), target)
target = targets[2]

assert np.allclose(multi_rail_encoding_state(3), target)
