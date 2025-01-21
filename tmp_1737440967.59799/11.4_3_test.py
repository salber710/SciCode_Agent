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



# Background: In quantum mechanics, a quantum channel is a mathematical model that describes the evolution of quantum states,
# particularly in the presence of noise or other environmental interactions. A common representation of quantum channels is through
# Kraus operators, which are a set of matrices that describe how a quantum state transforms under the channel. The action of a 
# quantum channel on a density matrix ρ can be expressed as ρ' = Σ_k K_k ρ K_k†, where K_k are the Kraus operators and K_k† is 
# the conjugate transpose of K_k. When applying a quantum channel to subsystems of a composite state, it is often necessary to 
# perform a partial trace over unaffected subsystems and use tensor product operations to ensure the channel acts only on the 
# intended subsystems. This process often requires reshaping the density matrix to properly reflect the dimensions of the subsystems.

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


    if sys is None and dim is None:
        # Apply the channel to the entire system
        new_rho = np.zeros_like(rho, dtype=np.float64)
        for K_op in K:
            new_rho += K_op @ rho @ K_op.T
        return new_rho

    elif sys is not None and dim is not None:
        # Apply the channel to specific subsystems
        full_dim = np.prod(dim)
        new_rho = np.zeros((full_dim, full_dim), dtype=np.float64)

        # Find the order of indices for the subsystems and reshape rho
        rho_reshaped = rho.reshape(dim + dim)
        
        for indices in itertools.product(*(range(d) for d in dim)):
            # Construct the indices for the systems that are being acted upon
            sys_indices = tuple(indices[i] for i in sys)
            non_sys_indices = tuple(indices[i] for i in range(len(dim)) if i not in sys)
            
            # Reshape the affected subsystem
            affected_subsystem = rho_reshaped[sys_indices + sys_indices]
            affected_subsystem = affected_subsystem.reshape(np.prod([dim[i] for i in sys]), 
                                                            np.prod([dim[i] for i in sys]))
            
            # Apply the Kraus operators on the affected subsystem
            transformed_subsystem = np.zeros_like(affected_subsystem)
            for K_op in K:
                transformed_subsystem += K_op @ affected_subsystem @ K_op.T
            
            # Reshape back and place the result in the new_rho
            transformed_subsystem = transformed_subsystem.reshape([dim[i] for i in sys] * 2)
            new_rho[np.ix_(indices, indices)] += transformed_subsystem

        return new_rho
    
    else:
        raise ValueError("Both sys and dim should be either lists or None")

from scicode.parse.parse import process_hdf5_to_tuple
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
