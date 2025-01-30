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



def multi_rail_encoding_state(rails):
    '''Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    '''
    # Dimension of each subsystem
    dim = 2 ** rails
    
    # Create a maximally entangled state using a Hadamard-like approach
    entangled_state = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            entangled_state[i, j] = 1 if (i ^ j) == 0 else 0
    
    # Flatten the matrix to create the state vector
    entangled_state = entangled_state.flatten()
    
    # Normalize the state vector
    entangled_state /= np.sqrt(dim)
    
    # Create the density matrix by taking the outer product of the state vector with itself
    state = np.outer(entangled_state, entangled_state)
    
    return state


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''


    # Compute the tensor product using a loop, but process in pairs from end to start
    if not args:
        return np.array([[]])  # Return an empty array if no arguments are provided

    # Start with the last matrix/vector
    M = args[-1]

    # Process in pairs
    for i in range(len(args) - 2, -1, -2):
        if i - 1 >= 0:
            M = np.kron(np.kron(args[i-1], args[i]), M)
        else:
            M = np.kron(args[i], M)

    return M



# Background: In quantum mechanics, a quantum channel is a mathematical model for noise and other processes that affect quantum states. 
# It is represented by a set of Kraus operators {K_i} that act on a density matrix ρ to produce a new density matrix ρ'. 
# The transformation is given by ρ' = Σ_i K_i ρ K_i†, where K_i† is the conjugate transpose of K_i. 
# When applying a quantum channel to specific subsystems of a composite quantum system, we need to consider the tensor product structure of the state. 
# The tensor function can be used to construct the appropriate operators that act on the entire system by embedding the Kraus operators into the larger space.

def apply_channel(K, rho, sys=None, dim=None):
    '''Applies the channel with Kraus operators in K to the state rho on
    systems specified by the list sys. The dimensions of the subsystems of
    rho are given by dim.
    Inputs:
    K: list of 2d array of floats, list of Kraus operators
    rho: 2d array of floats, input density matrix
    sys: list of int or None, list of subsystems to apply the channel, None means full system
    dim: list of int or None, list of dimensions of each subsystem, None means full system
    Output:
    matrix: output density matrix of floats
    '''




    def tensor(*args):
        '''Takes the tensor product of an arbitrary number of matrices/vectors.'''
        if not args:
            return np.array([[]])  # Return an empty array if no arguments are provided
        M = args[-1]
        for i in range(len(args) - 2, -1, -2):
            if i - 1 >= 0:
                M = np.kron(np.kron(args[i-1], args[i]), M)
            else:
                M = np.kron(args[i], M)
        return M

    if sys is None or dim is None:
        # Apply the channel to the entire system
        new_rho = np.zeros_like(rho)
        for k in K:
            new_rho += k @ rho @ k.conj().T
        return new_rho

    # Apply the channel to specified subsystems
    total_dim = np.prod(dim)
    new_rho = np.zeros((total_dim, total_dim), dtype=np.float64)

    # Iterate over all possible indices for the subsystems
    for indices in itertools.product(*[range(d) for d in dim]):
        # Create the identity operators for subsystems not in sys
        operators = [np.eye(d) if i not in sys else None for i, d in enumerate(dim)]

        # Apply the Kraus operators to the specified subsystems
        for k in K:
            for i, s in enumerate(sys):
                operators[s] = k if i == 0 else np.eye(dim[s])

            # Construct the full operator using the tensor product
            full_operator = tensor(*operators)

            # Apply the operator to the state
            new_rho += full_operator @ rho @ full_operator.conj().T

    return new_rho


try:
    targets = process_hdf5_to_tuple('11.4', 3)
    target = targets[0]
    K = [np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]])]
    rho = np.ones((2,2))/2
    assert np.allclose(apply_channel(K, rho, sys=None, dim=None), target)

    target = targets[1]
    K = [np.sqrt(0.8)*np.eye(2),np.sqrt(0.2)*np.array([[0,1],[1,0]])]
    rho = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    assert np.allclose(apply_channel(K, rho, sys=[2], dim=[2,2]), target)

    target = targets[2]
    K = [np.sqrt(0.8)*np.eye(2),np.sqrt(0.2)*np.array([[0,1],[1,0]])]
    rho = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    assert np.allclose(apply_channel(K, rho, sys=[1,2], dim=[2,2]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e