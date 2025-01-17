import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm

# Background: In quantum mechanics, rotation operators are used to describe rotations in the state space of a quantum system. 
# The rotation operators around the x, y, and z axes are denoted as Rx, Ry, and Rz, respectively. These operators are 
# represented by 2x2 matrices and are parameterized by an angle θ. The matrices are defined as follows:
# 
# Rx(θ) = [[cos(θ/2), -i*sin(θ/2)],
#          [-i*sin(θ/2), cos(θ/2)]]
#
# Ry(θ) = [[cos(θ/2), -sin(θ/2)],
#          [sin(θ/2), cos(θ/2)]]
#
# Rz(θ) = [[exp(-i*θ/2), 0],
#          [0, exp(i*θ/2)]]
#
# These matrices are used to rotate quantum states around the respective axes by the angle θ.


def rotation_matrices(axis, theta):
    '''Create rotation matrices Rx, Ry, and Rz with the given angle theta.
    Inputs:
    axis : int
        The rotation axis. 1 = x, 2 = y, 3 = z.
    theta : float
        The rotation angle.
    Output:
    R : matrix of shape(2, 2)
        The rotation matrix.
    '''
    if axis == 1:  # Rx
        R = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                      [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    elif axis == 2:  # Ry
        R = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                      [np.sin(theta / 2), np.cos(theta / 2)]])
    elif axis == 3:  # Rz
        R = np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]])
    else:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z).")
    
    return R



# Background: In quantum computing, the Unitary Coupled Cluster (UCC) ansatz is a method used to approximate the wavefunction of a quantum system. 
# It is particularly useful in quantum chemistry for simulating molecular systems. The UCC ansatz involves applying a series of unitary transformations 
# to an initial state, often the Hartree-Fock state, to generate a trial wavefunction. In this problem, we are tasked with creating a trial wavefunction 
# using the UCC ansatz with a single parameter θ. The specific transformation involves the operator exp(-i θ Y1 X2), where Y1 and X2 are Pauli operators 
# acting on the first and second qubits, respectively. The initial state is |01⟩, which is a two-qubit state. The task is to express this transformation 
# as a series of quantum gates acting on the initial state |00⟩.



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
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    # Create the initial state |00⟩
    initial_state = np.array([1, 0, 0, 0], dtype=complex).reshape(4, 1)

    # Create the |01⟩ state from |00⟩ using an X gate on the second qubit
    X2 = np.kron(I, X)
    state_01 = X2 @ initial_state

    # Construct the operator exp(-i θ Y1 X2)
    Y1 = np.kron(Y, I)
    X2 = np.kron(I, X)
    operator = expm(-1j * theta * Y1 @ X2)

    # Apply the operator to the |01⟩ state
    ansatz = operator @ state_01

    return ansatz


from scicode.parse.parse import process_hdf5_to_tuple

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
