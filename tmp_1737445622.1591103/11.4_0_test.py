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


# Background: In linear algebra and quantum mechanics, the tensor product (also known as the Kronecker product when applied to matrices)
# is an operation on two matrices (or vectors) of arbitrary size resulting in a block matrix. If A is an m×n matrix and B is a p×q
# matrix, then the Kronecker product A⊗B is an mp×nq matrix. In quantum mechanics, the tensor product is used to combine quantum states
# of different systems into a joint state, representing the combined system. This operation is essential when dealing with multi-particle
# systems or composite quantum systems. In this context, the tensor product allows for the representation of states in a higher-dimensional
# Hilbert space that encompasses the states of all subsystems.

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''



    # Start with the first element as the result
    M = args[0]

    # Iteratively compute the Kronecker product with the rest of the matrices/vectors
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M



# Background: In quantum mechanics, a quantum channel is a mathematical model used to describe the evolution of quantum states,
# particularly in open quantum systems where the system interacts with an environment. The evolution of a quantum state under a channel
# is often represented using Kraus operators, which are a set of matrices {K_i} that satisfy a completeness relation. The action of a
# quantum channel on a density matrix ρ is given by the transformation ρ' = Σ_i K_i ρ K_i†, where K_i† is the conjugate transpose of K_i.
# When dealing with composite systems, it is often necessary to apply the channel to specific subsystems, which requires considering
# the tensor product structure and appropriately embedding the Kraus operators into the larger Hilbert space.




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
    
    # If sys and dim are None, apply the channel to the entire system
    if sys is None and dim is None:
        # Apply the channel to the full density matrix
        new_rho = np.zeros_like(rho, dtype=np.float64)
        for kraus_operator in K:
            new_rho += kraus_operator @ rho @ kraus_operator.conj().T
        return new_rho
    
    # Otherwise, apply the channel to selected subsystems
    elif sys is not None and dim is not None:
        total_dim = np.prod(dim)  # Total dimension of the composite system
        new_rho = np.zeros_like(rho, dtype=np.float64)

        # Create the identity operators for unaffected subsystems
        identity_operators = [np.eye(d) for d in dim]

        # Iterate over all Kraus operators
        for kraus_operator in K:
            # Prepare the operator to apply to the entire system
            operator_list = [identity_operators[i] for i in range(len(dim))]
            
            # Replace identity with kraus_operator for the selected subsystems
            for i, subsystem in enumerate(sys):
                operator_list[subsystem] = kraus_operator

            # Calculate the tensor product of operators
            full_operator = operator_list[0]
            for op in operator_list[1:]:
                full_operator = np.kron(full_operator, op)

            # Apply the full operator to the density matrix
            new_rho += full_operator @ rho @ full_operator.conj().T
        
        return new_rho
    
    else:
        raise ValueError("Both sys and dim must be specified if one is provided.")


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

