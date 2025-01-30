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


# Background: In quantum mechanics, a composite quantum system can be described by a density matrix that represents the state of the system. 
# This system can be composed of multiple subsystems, each with its own dimension. Sometimes, it is necessary to permute the order of these 
# subsystems, which involves rearranging the dimensions of the density matrix according to a specified permutation. This operation is crucial 
# for tasks such as changing the basis of a quantum state or preparing a state for a specific quantum operation. The permutation of subsystems 
# is achieved by reshaping the density matrix into a multi-dimensional array, permuting the axes according to the desired order, and then 
# reshaping it back into a matrix form.


def syspermute(X, perm, dim):
    '''Permutes order of subsystems in the multipartite operator X.
    Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    perm: list of int containing the desired order
    dim: list of int containing the dimensions of all subsystems.
    Output:
    Y: 2d array of floats with equal dimensions, the density matrix of the permuted state
    '''
    # Calculate the number of subsystems
    num_subsystems = len(dim)
    
    # Ensure the permutation is valid
    if sorted(perm) != list(range(num_subsystems)):
        raise ValueError("Invalid permutation order.")
    
    # Check for non-integer elements in perm and dim
    if not all(isinstance(p, int) for p in perm):
        raise TypeError("All elements in perm must be integers.")
    if not all(isinstance(d, int) for d in dim):
        raise TypeError("All elements in dim must be integers.")
    
    # Check for negative or zero dimensions
    if any(d <= 0 for d in dim):
        raise ValueError("Dimensions must be positive integers.")
    
    # Reshape X into a multi-dimensional array
    try:
        reshaped_X = np.reshape(X, dim + dim)
    except ValueError:
        raise ValueError("Cannot reshape array to these dimensions.")
    
    # Create the permutation for the axes
    permuted_axes = perm + [p + num_subsystems for p in perm]
    
    # Permute the axes of the reshaped array
    permuted_X = np.transpose(reshaped_X, permuted_axes)
    
    # Reshape back to a 2D matrix
    Y = np.reshape(permuted_X, (np.prod(dim), np.prod(dim)))
    
    return Y



# Background: In quantum mechanics, the partial trace is an operation used to trace out (or discard) certain subsystems of a composite quantum state, 
# resulting in a reduced density matrix that describes the remaining subsystems. This is useful when we are interested in the state of a subsystem 
# without considering the rest of the system. Mathematically, if a composite system is described by a density matrix, the partial trace over a 
# subsystem is obtained by summing over the degrees of freedom of that subsystem. For a state with multiple subsystems, the partial trace can be 
# computed by reshaping the density matrix into a higher-dimensional array, permuting the axes to bring the traced-out subsystems together, 
# and then summing over those axes.

def partial_trace(X, sys, dim):
    '''Inputs:
    X: 2d array of floats with equal dimensions, the density matrix of the state
    sys: list of int containing systems over which to take the partial trace (i.e., the systems to discard).
    dim: list of int containing dimensions of all subsystems.
    Output:
    2d array of floats with equal dimensions, density matrix after partial trace.
    '''


    # Calculate the number of subsystems
    num_subsystems = len(dim)
    
    # Determine the subsystems to keep
    keep = [i for i in range(num_subsystems) if i not in sys]
    
    # Permute the subsystems so that the ones to be traced out are at the end
    perm = keep + sys
    
    # Use syspermute to rearrange the subsystems
    permuted_X = syspermute(X, perm, dim)
    
    # Calculate the dimensions of the subsystems to keep and trace out
    dim_keep = [dim[i] for i in keep]
    dim_trace = [dim[i] for i in sys]
    
    # Reshape the permuted matrix into a multi-dimensional array
    reshaped_X = np.reshape(permuted_X, dim_keep + dim_trace + dim_keep + dim_trace)
    
    # Sum over the axes corresponding to the traced-out subsystems
    for i in range(len(sys)):
        reshaped_X = np.trace(reshaped_X, axis1=len(dim_keep) + i, axis2=len(dim_keep) + len(dim_trace) + i)
    
    # Reshape back to a 2D matrix
    result_dim = np.prod(dim_keep)
    Y = np.reshape(reshaped_X, (result_dim, result_dim))
    
    return Y

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('71.5', 3)
target = targets[0]

X = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
assert np.allclose(partial_trace(X, [2], [2,2]), target)
target = targets[1]

X = np.kron(np.array([[1,0,0],[0,0,0],[0,0,0]]),np.array([[0,0],[0,1]]))
assert np.allclose(partial_trace(X, [2], [3,2]), target)
target = targets[2]

X = np.eye(6)/6
assert np.allclose(partial_trace(X, [1], [3,2]), target)
