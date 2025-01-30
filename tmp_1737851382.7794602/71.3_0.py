import numpy as np 
from scipy.optimize import fminbound
import itertools
from scipy.linalg import logm

# Background: In quantum mechanics, a ket vector |j⟩ in a d-dimensional space is a column vector with a 1 in the j-th position and 0s elsewhere. 
# This is a standard basis vector in the context of quantum states. When dealing with multiple quantum systems, the tensor product of individual 
# kets is used to represent the combined state. The tensor product of vectors results in a higher-dimensional vector space, where the dimensions 
# are the product of the individual dimensions. In this problem, we need to construct such a ket vector or a tensor product of multiple ket vectors 
# based on the input dimensions and indices.


def ket(dim, *args):
    '''Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    '''
    if isinstance(dim, int):
        # Single dimension and single index
        if not isinstance(args[0], int):
            raise TypeError("For single dimension, index must be an integer.")
        j = args[0]
        if j < 0 or j >= dim:
            raise IndexError("Index out of bounds.")
        out = np.zeros(dim)
        out[j] = 1.0
    elif isinstance(dim, list) and isinstance(args[0], list):
        # Multiple dimensions and multiple indices
        dims = dim
        indices = args[0]
        if len(dims) != len(indices):
            raise ValueError("Dimensions and indices must have the same length.")
        if not dims or not indices:
            raise ValueError("Dimensions and indices cannot be empty.")
        # Validate dimensions are positive and indices are within bounds
        for d, idx in zip(dims, indices):
            if d <= 0:
                raise ValueError("Dimensions must be positive integers.")
            if idx < 0 or idx >= d:
                raise IndexError("Index out of bounds.")
        # Start with the first ket
        out = np.zeros(dims[0])
        out[indices[0]] = 1.0
        # Tensor product with subsequent kets
        for d, j in zip(dims[1:], indices[1:]):
            ket_j = np.zeros(d)
            ket_j[j] = 1.0
            out = np.kron(out, ket_j)
    else:
        raise ValueError("Invalid input format for dim and args.")
    
    return out


# Background: In linear algebra and quantum mechanics, the tensor product (also known as the Kronecker product) is an operation on two matrices or vectors that results in a block matrix. For vectors, the tensor product results in a higher-dimensional vector. For matrices, it results in a larger matrix that combines the information of the input matrices. The tensor product is essential in quantum mechanics for describing the state of a composite quantum system. The Kronecker product of matrices A (of size m x n) and B (of size p x q) is a matrix of size (m*p) x (n*q).


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one input matrix/vector is required.")
    
    # Start with the first matrix/vector
    M = args[0]
    
    # Iterate over the remaining matrices/vectors and compute the Kronecker product
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M



# Background: In quantum mechanics, a quantum channel is a mathematical model for the physical process of transmitting quantum states. 
# The action of a quantum channel on a quantum state can be described using Kraus operators. The Kraus representation of a quantum channel 
# is given by the equation: 𝒩(ρ) = ∑_i K_i ρ K_i^†, where K_i are the Kraus operators and K_i^† is the conjugate transpose of K_i. 
# The Kraus operators satisfy the completeness relation ∑_i K_i^† K_i = I, where I is the identity operator. 
# When a quantum channel acts on a specific subsystem of a composite quantum state, the Kraus operators are applied to that subsystem, 
# while the identity operator acts on the other subsystems. This is achieved by taking the tensor product of the identity operators 
# and the Kraus operators, ensuring that the channel acts only on the specified subsystem.



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
        dim = [rho.shape[0]]
        sys = [0]

    # Calculate the total dimension of the system
    total_dim = np.prod(dim)

    # Initialize the output density matrix
    output_rho = np.zeros((total_dim, total_dim), dtype=complex)

    # Iterate over each Kraus operator
    for K_i in K:
        # Construct the full operator for the system
        full_operator = 1
        for i in range(len(dim)):
            if i in sys:
                # Apply the Kraus operator to the specified subsystem
                full_operator = np.kron(full_operator, K_i)
            else:
                # Apply the identity operator to the other subsystems
                full_operator = np.kron(full_operator, np.eye(dim[i]))

        # Apply the channel to the state rho
        output_rho += full_operator @ rho @ full_operator.conj().T

    return output_rho

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
