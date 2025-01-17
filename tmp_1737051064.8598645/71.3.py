import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

# Background: In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is a column vector with a 1 in the j-th position and 0s elsewhere. 
# This is a standard basis vector in the context of quantum states. When dealing with multiple quantum systems, the tensor product of individual 
# kets is used to represent the combined state. The tensor product of vectors is a way to construct a new vector space from two or more vector spaces. 
# If j is a list, it represents multiple indices for which we need to create a tensor product of basis vectors. Similarly, if d is a list, it 
# represents the dimensions of each corresponding basis vector.


def ket(dim, args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single dimension and single index
        out = np.zeros(dim)
        out[args] = 1.0
    else:
        # Multiple dimensions and indices
        vectors = []
        for d, j in zip(dim, args):
            vec = np.zeros(d)
            vec[j] = 1.0
            vectors.append(vec)
        # Compute the tensor product of all vectors
        out = vectors[0]
        for vec in vectors[1:]:
            out = np.kron(out, vec)
    
    return out


# Background: In linear algebra, the tensor product (also known as the Kronecker product) of two matrices is a way to construct a new matrix from two given matrices. 
# The tensor product of matrices is a generalization of the outer product of vectors. If A is an m×n matrix and B is a p×q matrix, 
# then their tensor product A ⊗ B is an mp×nq matrix. This operation is widely used in quantum mechanics to describe the state of a composite quantum system. 
# The tensor product of multiple matrices is computed by iteratively applying the Kronecker product to pairs of matrices.

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''

    
    if not args:
        raise ValueError("At least one matrix/vector is required for the tensor product.")
    
    # Start with the first matrix/vector
    M = args[0]
    
    # Iteratively compute the tensor product with the remaining matrices/vectors
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M



# Background: In quantum mechanics, a quantum channel is a mathematical model for noise and other processes affecting quantum states. 
# The action of a quantum channel on a quantum state can be described using Kraus operators. The Kraus representation of a quantum channel 
# is given by: $\mathcal{N}(\rho) = \sum_i K_i \rho K_i^\dagger$, where $K_i$ are the Kraus operators and $\rho$ is the density matrix of the state. 
# The Kraus operators satisfy the completeness relation $\sum_i K_i^\dagger K_i = \mathbb{I}$. When a channel acts on a specific subsystem of a 
# composite quantum state, the Kraus operators are applied to that subsystem while the identity operator acts on the other subsystems. 
# This is achieved by constructing the operator as a tensor product of identity operators and the Kraus operator, appropriately placed.


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
        result = np.zeros_like(rho)
        for Ki in K:
            result += Ki @ rho @ Ki.conj().T
        return result
    
    # Apply the channel to specified subsystems
    num_subsystems = len(dim)
    identity_operators = [np.eye(d) for d in dim]
    
    # Initialize the result density matrix
    result = np.zeros_like(rho)
    
    for Ki in K:
        # Construct the full operator with Ki acting on the specified subsystem
        operators = identity_operators.copy()
        for i, subsystem in enumerate(sys):
            operators[subsystem] = Ki if i == 0 else np.eye(dim[subsystem])
        
        # Compute the tensor product of the operators
        full_operator = operators[0]
        for op in operators[1:]:
            full_operator = np.kron(full_operator, op)
        
        # Apply the operator to the density matrix
        result += full_operator @ rho @ full_operator.conj().T
    
    return result


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('71.3', 3)
target = targets[0]

K = [np.eye(2)]
rho = np.array([[0.8,0],[0,0.2]])
assert np.allclose(apply_channel(K, rho, sys=None, dim=None), target)
target = targets[1]

K = [np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]])]
rho = np.ones((2,2))/2
assert np.allclose(apply_channel(K, rho, sys=None, dim=None), target)
target = targets[2]

K = [np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]])]
rho = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
assert np.allclose(apply_channel(K, rho, sys=[2], dim=[2,2]), target)
