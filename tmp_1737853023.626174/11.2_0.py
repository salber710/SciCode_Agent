import numpy as np
import itertools
import scipy.linalg

# Background: In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is a column vector with a 1 in the j-th position and 0s elsewhere. 
# This is known as a standard basis vector. When dealing with multiple quantum systems, the state of the combined system is represented by the 
# tensor product of the individual states. If j is a list, it represents multiple indices for which we need to create a tensor product of 
# standard basis vectors. If d is a list, it specifies the dimensions of each individual space for the tensor product.




def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single space case
        d = dim
        j = args
        if not isinstance(j, int):
            raise ValueError("Index must be an integer for single dimension.")
        if j < 0 or j >= d:
            raise IndexError("Index out of bounds for the dimension specified.")
        # Create a d-dimensional basis vector with 1 at position j
        out = np.zeros(d)
        out[j] = 1.0
    else:
        # Multiple spaces case
        d_list = dim
        j_list = args
        if not isinstance(j_list, list):
            raise ValueError("Indices must be a list for multiple dimensions.")
        if len(d_list) != len(j_list):
            raise ValueError("Dimensions and indices lengths do not match.")
        # Create the tensor product of basis vectors
        basis_vectors = []
        for d, j in zip(d_list, j_list):
            if j < 0 or j >= d:
                raise IndexError("Index out of bounds for the dimension specified.")
            vec = np.zeros(d)
            vec[j] = 1.0
            basis_vectors.append(vec)
        # Compute the tensor product
        out = basis_vectors[0]
        for vec in basis_vectors[1:]:
            out = np.kron(out, vec)
    
    return out



# Background: In quantum mechanics, a bipartite maximally entangled state is a quantum state of two subsystems that cannot be described independently.
# One common example is the Bell state, which is a specific type of entangled state. In the context of multi-rail encoding, each party's state is
# represented across multiple "rails" or dimensions. The goal is to create a density matrix representing a maximally entangled state where each
# subsystem is encoded using m rails. The density matrix is a square matrix that describes the statistical state of a quantum system, and for a
# maximally entangled state, it will have specific properties that reflect the entanglement between the subsystems.




def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    # Total dimension for each subsystem
    dim = 2 ** rails
    
    # Create the maximally entangled state |Φ⟩ = (1/√dim) * Σ |i⟩|i⟩
    # where i ranges over all possible states of the subsystem
    state_vector = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        # Create the basis vector |i⟩ for the first subsystem
        ket_i = np.zeros(dim)
        ket_i[i] = 1.0
        
        # Create the basis vector |i⟩ for the second subsystem
        ket_j = np.zeros(dim)
        ket_j[i] = 1.0
        
        # Tensor product |i⟩|i⟩
        state_vector += np.outer(ket_i, ket_j)
    
    # Normalize the state vector
    state_vector /= np.sqrt(dim)
    
    # Create the density matrix ρ = |Φ⟩⟨Φ|
    state = np.outer(state_vector.flatten(), state_vector.flatten().conj())
    
    return state

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('11.2', 3)
target = targets[0]

assert np.allclose(multi_rail_encoding_state(1), target)
target = targets[1]

assert np.allclose(multi_rail_encoding_state(2), target)
target = targets[2]

assert np.allclose(multi_rail_encoding_state(3), target)
