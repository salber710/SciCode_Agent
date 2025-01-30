from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg

def ket(dim, args):


    def create_basis_vector(d, i):
        # Create a d-dimensional zero vector and set the i-th position to 1
        vec = np.zeros(d)
        vec[i] = 1
        return vec

    if isinstance(args, list):
        # Compute the tensor product of basis vectors for each dimension and index in args
        result = create_basis_vector(dim[0], args[0])
        for d, i in zip(dim[1:], args[1:]):
            result = np.kron(result, create_basis_vector(d, i))
    else:
        # Single dimension and index
        result = create_basis_vector(dim, args)

    return result



# Background: In quantum mechanics, a bipartite maximally entangled state is a state of two subsystems where the state of each subsystem is completely undetermined, but the two subsystems are perfectly correlated. For a system with m-rail encoding, each subsystem is represented by m qubits, and the entangled state is a superposition of all possible states where the two subsystems are in the same state. The density matrix of such a state is given by the outer product of the state vector with itself. The state vector for a maximally entangled state in m-rail encoding can be constructed using the tensor product of basis vectors, where each basis vector corresponds to a computational basis state.




def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    # Number of basis states for each subsystem
    dim = 2 ** rails
    
    # Create the maximally entangled state vector
    entangled_state = np.zeros((dim * dim,), dtype=np.float64)
    
    # Iterate over all possible states for one subsystem
    for i in range(dim):
        # Create the basis vector |i> for the first subsystem
        basis_vector_i = np.zeros((dim,), dtype=np.float64)
        basis_vector_i[i] = 1
        
        # Create the basis vector |i> for the second subsystem
        basis_vector_j = np.zeros((dim,), dtype=np.float64)
        basis_vector_j[i] = 1
        
        # Compute the tensor product |i>|i> and add to the entangled state
        entangled_state += np.kron(basis_vector_i, basis_vector_j)
    
    # Normalize the entangled state
    entangled_state /= np.sqrt(dim)
    
    # Compute the density matrix as the outer product of the entangled state with itself
    state = np.outer(entangled_state, entangled_state)
    
    return state


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