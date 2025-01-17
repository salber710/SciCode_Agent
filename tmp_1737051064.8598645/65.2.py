import numpy as np
from scipy.linalg import sqrtm
import itertools

# Background: The tensor product, also known as the Kronecker product, is an operation on two matrices (or vectors) that results in a block matrix. 
# If A is an m x n matrix and B is a p x q matrix, their Kronecker product A ⊗ B is an mp x nq matrix. 
# The Kronecker product is a generalization of the outer product from vectors to matrices. 
# It is used in various fields such as quantum computing, signal processing, and the study of multi-linear algebra. 
# In this function, we aim to compute the tensor product of an arbitrary number of matrices or vectors.


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
    
    # Iterate over the remaining matrices/vectors and compute the Kronecker product
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    
    return M



# Background: In quantum mechanics, a quantum channel is a mathematical model for the physical process of transmitting quantum states. 
# It is represented by a set of Kraus operators {K_i}, which are used to describe the evolution of a quantum state ρ. 
# The action of a quantum channel on a state ρ is given by ρ' = Σ_i K_i ρ K_i†, where K_i† is the conjugate transpose of K_i. 
# When applying a quantum channel to specific subsystems of a composite quantum system, we need to consider the tensor product structure of the state. 
# The tensor function can be used to construct the appropriate operators for the subsystems. 
# If the channel is applied to the entire system, the dimensions of the subsystems are not needed. 
# However, if the channel is applied to specific subsystems, the dimensions of these subsystems must be specified to correctly apply the channel.



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
        for k in K:
            new_rho += k @ rho @ k.conj().T
        return new_rho
    else:
        # Apply the channel to specified subsystems
        total_dim = np.prod(dim)
        if rho.shape != (total_dim, total_dim):
            raise ValueError("The dimensions of rho do not match the product of the subsystem dimensions.")
        
        # Create identity operators for subsystems not in sys
        identity_ops = [np.eye(d) for d in dim]
        
        # Initialize the new density matrix
        new_rho = np.zeros_like(rho)
        
        # Iterate over all Kraus operators
        for k in K:
            # Construct the full operator for the entire system
            full_operator = [identity_ops[i] if i not in sys else k for i in range(len(dim))]
            full_operator = np.kron.reduce(full_operator)
            
            # Apply the operator to the state
            new_rho += full_operator @ rho @ full_operator.conj().T
        
        return new_rho


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
