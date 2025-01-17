import numpy as np
import itertools
import scipy.linalg
def ket(dim, args):
    """Input:
    dim: int or list, dimension of the ket
    args: int or list, the i-th basis vector
    Output:
    out: dim dimensional array of float, the matrix representation of the ket
    """
    if isinstance(dim, int):
        out = np.zeros(dim)
        out[args] = 1.0
    else:
        vectors = []
        for (d, j) in zip(dim, args):
            vec = np.zeros(d)
            vec[j] = 1.0
            vectors.append(vec)
        out = vectors[0]
        for vec in vectors[1:]:
            out = np.kron(out, vec)
    return out
def multi_rail_encoding_state(rails):
    """Returns the density matrix of the multi-rail encoding state
    Input:
    rails: int, number of rails
    Output:
    state: 2**(2*rails) x 2**(2*rails) dimensional array of numpy.float64 type
    """
    dim = 2 ** rails
    entangled_state = np.zeros((dim * dim,), dtype=np.float64)
    for i in range(dim):
        ket_i = ket(dim, i)
        ket_j = ket(dim, i)
        tensor_product = np.kron(ket_i, ket_j)
        entangled_state += tensor_product
    entangled_state /= np.sqrt(dim)
    state = np.outer(entangled_state, entangled_state)
    return state
def tensor(*args):
    """Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of nd arrays of floats, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices, 2d array of floats
    """
    M = args[0]
    for matrix in args[1:]:
        M = np.kron(M, matrix)
    return M


# Background: In quantum mechanics, a quantum channel is a mathematical model for noise and other processes that affect quantum states. 
# It is represented by a set of Kraus operators {K_i} that act on a density matrix ρ, transforming it according to the rule:
# ρ' = Σ_i K_i ρ K_i†, where K_i† is the conjugate transpose of K_i. 
# When applying a quantum channel to specific subsystems of a composite quantum system, we need to consider the tensor product structure of the state.
# The function should apply the channel to specified subsystems, which may involve reshaping and permuting the state to isolate the subsystems, 
# applying the channel, and then reconstructing the full state.




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
        rho = rho.reshape([total_dim, total_dim])
        
        # Create a list of identity matrices for subsystems not in sys
        id_matrices = [np.eye(d) for d in dim]
        
        # Initialize the new density matrix
        new_rho = np.zeros_like(rho)
        
        # Iterate over all combinations of Kraus operators
        for kraus_ops in itertools.product(K, repeat=len(sys)):
            # Start with the identity matrix
            op = np.eye(total_dim)
            
            # Apply the Kraus operators to the specified subsystems
            for idx, s in enumerate(sys):
                # Create the operator for this subsystem
                op_s = np.eye(dim[s])
                op_s = kraus_ops[idx] @ op_s @ kraus_ops[idx].conj().T
                
                # Tensor product with identity matrices for other subsystems
                op = np.kron(op, op_s)
            
            # Apply the operator to the density matrix
            new_rho += op @ rho @ op.conj().T
        
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
