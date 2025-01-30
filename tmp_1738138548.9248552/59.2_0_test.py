from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm


def rotation_matrices(axis, theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    if axis == 1:  # Rx
        R = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex_)
    elif axis == 2:  # Ry
        R = np.array([[c, -s], [s, c]], dtype=np.complex_)
    elif axis == 3:  # Rz
        R = np.array([[np.cos(-theta / 2) + 1j * np.sin(-theta / 2), 0], 
                      [0, np.cos(theta / 2) + 1j * np.sin(theta / 2)]], dtype=np.complex_)
    else:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z).")
    return R



# Background: In quantum computing, the Unitary Coupled Cluster (UCC) ansatz is a method used to approximate the wavefunction of a quantum system. 
# The UCC ansatz is often used in quantum chemistry to simulate molecular systems. The ansatz wavefunction is constructed by applying a series of 
# quantum gates to an initial state. In this problem, we are tasked with creating a trial wavefunction using the UCC ansatz with a single parameter 
# theta. The specific form of the ansatz is given by the expression: 
# |\psi(\theta)\rangle = exp(-i * theta * Y1 * X2) |01\rangle, 
# where Y1 and X2 are Pauli-Y and Pauli-X operators acting on the first and second qubits, respectively. The initial state is |01\rangle, which 
# corresponds to the second qubit being in the |1> state and the first qubit in the |0> state. The exponential of the operator is computed using 
# matrix exponentiation, which can be done using the expm function from scipy.linalg.



def create_ansatz(theta):
    '''Create the ansatz wavefunction with a given theta.
    Input:
    theta : float
        The only variational parameter.
    Output:
    ansatz : array of shape (4, 1)
        The ansatz wavefunction.
    '''
    # Define Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=np.complex_)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex_)
    
    # Create the operator Y1 * X2
    Y1_X2 = np.kron(Y, X)
    
    # Compute the unitary operator exp(-i * theta * Y1 * X2)
    U = expm(-1j * theta * Y1_X2)
    
    # Initial state |01> in the computational basis
    initial_state = np.array([0, 1, 0, 0], dtype=np.complex_).reshape(4, 1)
    
    # Apply the unitary operator to the initial state
    ansatz = U @ initial_state
    
    return ansatz


try:
    targets = process_hdf5_to_tuple('59.2', 3)
    target = targets[0]
    psi0 = np.kron([[1],[0]], [[1],[0]])
    I = np.array([[1, 0], [0, 1]])
    psi = np.dot(np.kron(I, rotation_matrices(1, np.pi)), psi0)
    theta = np.pi / 4
    Sx = np.array([[0, 1], [1, 0]])
    Sy = np.array([[0, -1j], [1j, 0]])
    ansatz_o = np.dot(expm(-1j * theta * np.kron(Sy, Sx)), psi)
    ansatz_c = create_ansatz(theta)
    assert (np.isclose(np.vdot(ansatz_o, ansatz_c), np.linalg.norm(ansatz_o) * np.linalg.norm(ansatz_c))) == target

    target = targets[1]
    psi0 = np.kron([[1],[0]], [[1],[0]])
    I = np.array([[1, 0], [0, 1]])
    psi = np.dot(np.kron(I, rotation_matrices(1, np.pi)), psi0)
    theta = np.pi / 8
    Sx = np.array([[0, 1], [1, 0]])
    Sy = np.array([[0, -1j], [1j, 0]])
    ansatz_o = np.dot(expm(-1j * theta * np.kron(Sy, Sx)), psi)
    ansatz_c = create_ansatz(theta)
    assert (np.isclose(np.vdot(ansatz_o, ansatz_c), np.linalg.norm(ansatz_o) * np.linalg.norm(ansatz_c))) == target

    target = targets[2]
    psi0 = np.kron([[1],[0]], [[1],[0]])
    I = np.array([[1, 0], [0, 1]])
    psi = np.dot(np.kron(I, rotation_matrices(1, np.pi)), psi0)
    theta = np.pi / 6
    Sx = np.array([[0, 1], [1, 0]])
    Sy = np.array([[0, -1j], [1j, 0]])
    ansatz_o = np.dot(expm(-1j * theta * np.kron(Sy, Sx)), psi)
    ansatz_c = create_ansatz(theta)
    assert (np.isclose(np.vdot(ansatz_o, ansatz_c), np.linalg.norm(ansatz_o) * np.linalg.norm(ansatz_c))) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e