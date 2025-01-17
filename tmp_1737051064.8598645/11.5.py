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
def apply_channel(K, rho, sys=None, dim=None):
    """Applies the channel with Kraus operators in K to the state rho on
    systems specified by the list sys. The dimensions of the subsystems of
    rho are given by dim.
    Inputs:
    K: list of 2d array of floats, list of Kraus operators
    rho: 2d array of floats, input density matrix
    sys: list of int or None, list of subsystems to apply the channel, None means full system
    dim: list of int or None, list of dimensions of each subsystem, None means full system
    Output:
    matrix: output density matrix of floats
    """
    if sys is None or dim is None:
        new_rho = np.zeros_like(rho)
        for k in K:
            new_rho += k @ rho @ k.conj().T
        return new_rho
    else:
        total_dim = np.prod(dim)
        rho = rho.reshape([total_dim, total_dim])
        id_matrices = [np.eye(d) for d in dim]
        new_rho = np.zeros_like(rho)
        for kraus_ops in itertools.product(K, repeat=len(sys)):
            op = np.eye(total_dim)
            for (idx, s) in enumerate(sys):
                op_s = np.eye(dim[s])
                op_s = kraus_ops[idx] @ op_s @ kraus_ops[idx].conj().T
                op = np.kron(op, op_s)
            new_rho += op @ rho @ op.conj().T
        return new_rho


# Background: The generalized amplitude damping channel is a quantum channel that models the interaction of a qubit with a thermal bath at non-zero temperature. It is characterized by two parameters: the damping parameter gamma, which represents the probability of energy dissipation, and the thermal parameter N, which represents the average number of excitations in the environment. The channel is described by four Kraus operators, which are 2x2 matrices that satisfy the completeness relation. These operators are used to describe the evolution of the quantum state under the influence of the channel.


def generalized_amplitude_damping_channel(gamma, N):
    '''Generates the generalized amplitude damping channel.
    Inputs:
    gamma: float, damping parameter
    N: float, thermal parameter
    Output:
    kraus: list of Kraus operators as 2x2 arrays of floats, [A1, A2, A3, A4]
    '''
    # Calculate the Kraus operators
    A1 = np.sqrt(gamma) * np.array([[1, 0], [0, np.sqrt(1 - N)]])
    A2 = np.sqrt(gamma) * np.array([[0, np.sqrt(N)], [0, 0]])
    A3 = np.sqrt(1 - gamma) * np.array([[np.sqrt(1 - N), 0], [0, 1]])
    A4 = np.sqrt(1 - gamma) * np.array([[0, 0], [np.sqrt(N), 0]])
    
    kraus = [A1, A2, A3, A4]
    return kraus


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('11.5', 3)
target = targets[0]

assert np.allclose(generalized_amplitude_damping_channel(0, 0), target)
target = targets[1]

assert np.allclose(generalized_amplitude_damping_channel(0.8, 0), target)
target = targets[2]

assert np.allclose(generalized_amplitude_damping_channel(0.5, 0.5), target)
