import numpy as np
from scipy.linalg import sqrtm
import itertools

# Background: The tensor product, also known as the Kronecker product, is an operation on two matrices of arbitrary size resulting in a block matrix. 
# It is a generalization of the outer product from vectors to matrices. The tensor product of two matrices A (of size m x n) and B (of size p x q) 
# is a matrix of size (m*p) x (n*q). The elements of the resulting matrix are computed as the product of each element of A with the entire matrix B. 
# This operation is associative, meaning the order of operations does not affect the final result, which allows us to extend it to an arbitrary number 
# of matrices or vectors. In the context of quantum mechanics and other fields, the tensor product is used to describe the state space of a composite 
# system as the product of the state spaces of its components.


def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    '''
    if not args:
        raise ValueError("At least one matrix or vector is required for the tensor product.")
    
    # Start with the first matrix/vector
    M = args[0]
    
    # Iterate over the remaining matrices/vectors and compute the tensor product
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M



# Background: In quantum mechanics, a quantum channel is a mathematical model for the physical process of transmitting quantum states. 
# It is represented by a set of Kraus operators, which are matrices that describe the effect of the channel on a quantum state. 
# The action of a quantum channel on a density matrix (quantum state) is given by the sum of transformations of the state by each Kraus operator. 
# Specifically, if K = {K1, K2, ..., Kn} is a set of Kraus operators, the channel transforms a density matrix ρ as: 
# ρ' = Σ_i (Ki * ρ * Ki†), where Ki† is the conjugate transpose of Ki. 
# When applying a channel to specific subsystems of a composite quantum system, we need to consider the tensor product structure of the state. 
# The function should apply the channel to specified subsystems, which requires reshaping and permuting the state according to the dimensions of the subsystems.



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
        new_rho = np.zeros_like(rho)
        for kraus in K:
            new_rho += kraus @ rho @ kraus.conj().T
        return new_rho
    else:
        # Apply the channel to specified subsystems
        total_dim = np.prod(dim)
        if rho.shape != (total_dim, total_dim):
            raise ValueError("The dimensions of rho do not match the product of the subsystem dimensions.")
        
        # Reshape rho into a multi-dimensional array
        rho_reshaped = rho.reshape(dim + dim)
        
        # Prepare the identity for unaffected subsystems
        unaffected_dims = [dim[i] for i in range(len(dim)) if i not in sys]
        identity = np.eye(np.prod(unaffected_dims))
        
        # Initialize the new rho
        new_rho = np.zeros_like(rho_reshaped)
        
        # Iterate over all combinations of indices for the unaffected subsystems
        for indices in itertools.product(*[range(d) for d in unaffected_dims]):
            # Construct the slice for the current indices
            slice_indices = tuple(slice(None) if i in sys else indices[unaffected_dims.index(dim[i])] for i in range(len(dim)))
            
            # Extract the relevant part of rho
            rho_part = rho_reshaped[slice_indices]
            
            # Apply the channel to this part
            new_rho_part = np.zeros_like(rho_part)
            for kraus in K:
                new_rho_part += kraus @ rho_part @ kraus.conj().T
            
            # Place the transformed part back into the new rho
            new_rho[slice_indices] = new_rho_part
        
        # Reshape back to the original 2D form
        return new_rho.reshape(total_dim, total_dim)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('65.2', 3)
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
