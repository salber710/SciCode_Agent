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
        out = np.zeros(dim)
        out[args] = 1.0
    else:
        # Multiple spaces case
        vectors = []
        for d, j in zip(dim, args):
            vec = np.zeros(d)
            vec[j] = 1.0
            vectors.append(vec)
        out = vectors[0]
        for vec in vectors[1:]:
            out = np.kron(out, vec)
    
    return out



# Background: In quantum mechanics, a bipartite maximally entangled state is a quantum state of two subsystems that cannot be described independently of each other. 
# For a system with m-rail encoding, each party (subsystem) is represented by m qubits. The maximally entangled state for two parties, each with m qubits, is 
# often represented as a superposition of states where each party has the same state. The density matrix of such a state is given by the outer product of the 
# state vector with itself. The state vector for a maximally entangled state can be constructed using the ket function to create basis vectors and then 
# summing over all possible states where both parties have the same state.




def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    # Number of qubits for each party
    dim = 2 ** rails
    
    # Initialize the state vector for the maximally entangled state
    entangled_state = np.zeros((dim * dim,), dtype=np.float64)
    
    # Iterate over all possible states for one party
    for i in range(dim):
        # Create the basis vector |i⟩ for the first party
        ket_i = ket(dim, i)
        
        # Create the basis vector |i⟩ for the second party
        ket_j = ket(dim, i)
        
        # Tensor product |i⟩|i⟩
        tensor_product = np.kron(ket_i, ket_j)
        
        # Add to the entangled state
        entangled_state += tensor_product
    
    # Normalize the state vector
    entangled_state /= np.sqrt(dim)
    
    # Compute the density matrix as the outer product of the state vector with itself
    state = np.outer(entangled_state, entangled_state)
    
    return state


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('11.2', 3)
target = targets[0]

assert np.allclose(multi_rail_encoding_state(1), target)
target = targets[1]

assert np.allclose(multi_rail_encoding_state(2), target)
target = targets[2]

assert np.allclose(multi_rail_encoding_state(3), target)
