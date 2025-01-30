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
    if not isinstance(rails, int):
        raise TypeError("Rails must be an integer.")
    if rails < 0:
        raise ValueError("Rails must be a non-negative integer.")
    
    # Total dimension for each subsystem
    dim = 2 ** rails
    
    # Create the maximally entangled state |Φ⟩ = (1/√dim) * Σ |i⟩|i⟩
    # where i ranges over all possible states of the subsystem
    state_vector = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        # Create the basis vector |i⟩ for the first subsystem
        ket_i = np.zeros(dim, dtype=np.complex128)
        ket_i[i] = 1.0
        
        # Create the basis vector |i⟩ for the second subsystem
        ket_j = np.zeros(dim, dtype=np.complex128)
        ket_j[i] = 1.0
        
        # Tensor product |i⟩|i⟩
        state_vector += np.outer(ket_i, ket_j)
    
    # Normalize the state vector
    state_vector /= np.sqrt(dim)
    
    # Create the density matrix ρ = |Φ⟩⟨Φ|
    state = np.outer(state_vector.flatten(), state_vector.flatten().conj())
    
    # Ensure the matrix is Hermitian and has non-negative eigenvalues
    state = (state + state.conj().T) / 2
    
    return state


# Background: In linear algebra and quantum mechanics, the tensor product (also known as the Kronecker product) is an operation on two matrices or vectors
# that results in a block matrix. For vectors, it results in a higher-dimensional vector. The tensor product of matrices A and B, denoted A ⊗ B, is a 
# matrix formed by multiplying each element of A by the entire matrix B. This operation is associative, meaning the order of operations does not affect 
# the result, allowing us to compute the tensor product of an arbitrary number of matrices or vectors sequentially.

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''


    if len(args) == 0:
        raise ValueError("At least one matrix or vector is required for the tensor product.")

    # Start with the first matrix/vector
    try:
        M = args[0].astype(np.float64)  # Ensure the first matrix/vector is of type float64
    except ValueError as e:
        raise TypeError("Input matrices must contain numeric values.") from e

    # Sequentially compute the tensor product with each subsequent matrix/vector
    for matrix in args[1:]:
        try:
            M = np.kron(M, matrix.astype(np.float64))  # Convert each matrix to float64 before the operation
        except ValueError as e:
            raise TypeError("Input matrices must contain numeric values.") from e

    return M



# Background: In quantum mechanics, a quantum channel is a mathematical model for the physical process of transmitting quantum states. 
# It is represented by a set of Kraus operators {K_i}, which are matrices that describe the effect of the channel on a quantum state. 
# The action of a quantum channel on a density matrix ρ is given by the transformation ρ' = Σ_i K_i ρ K_i†, where K_i† is the conjugate 
# transpose of K_i. When applying a channel to specific subsystems of a composite quantum system, we need to consider the tensor product 
# structure of the state and apply the channel only to the specified subsystems. This involves using the tensor product to construct the 
# full operator that acts on the entire system, while the identity operator acts on the subsystems not affected by the channel.

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




    if sys is None or dim is None:
        # Apply the channel to the entire system
        new_rho = np.zeros_like(rho, dtype=np.complex128)
        for K_i in K:
            new_rho += K_i @ rho @ K_i.conj().T
        return new_rho

    # Apply the channel to specified subsystems
    total_dim = np.prod(dim)
    identity = np.eye(total_dim, dtype=np.complex128)
    new_rho = np.zeros((total_dim, total_dim), dtype=np.complex128)

    # Generate all possible indices for the subsystems
    indices = list(itertools.product(*[range(d) for d in dim]))

    for idx in indices:
        # Create the basis vector |idx⟩
        ket_idx = np.zeros(total_dim, dtype=np.complex128)
        ket_idx[np.ravel_multi_index(idx, dim)] = 1.0

        # Create the projector |idx⟩⟨idx|
        projector = np.outer(ket_idx, ket_idx.conj())

        # Apply the channel to the specified subsystems
        for K_i in K:
            # Construct the full operator that acts on the entire system
            full_operator = identity.copy()
            for s in sys:
                # Apply the Kraus operator to the subsystem
                full_operator = np.kron(full_operator, K_i if s in sys else np.eye(dim[s], dtype=np.complex128))

            # Update the new density matrix
            new_rho += full_operator @ rho @ full_operator.conj().T

    return new_rho

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
