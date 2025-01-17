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



# Background: In linear algebra and quantum mechanics, the tensor product (also known as the Kronecker product) is an operation on two matrices or vectors that results in a block matrix. 
# The tensor product of two matrices A (of size m x n) and B (of size p x q) is a matrix of size (m*p) x (n*q). 
# This operation is crucial in quantum mechanics for describing the combined state of two or more quantum systems. 
# The tensor product is associative, meaning the order of operations does not affect the result, allowing us to extend it to an arbitrary number of matrices or vectors.

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''



    # Start with the first matrix/vector
    M = args[0]
    
    # Iteratively compute the tensor product with each subsequent matrix/vector
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('11.3', 3)
target = targets[0]

assert np.allclose(tensor([0,1],[0,1]), target)
target = targets[1]

assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)
target = targets[2]

assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)
