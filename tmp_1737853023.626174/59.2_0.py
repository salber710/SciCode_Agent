import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm

# Background: In quantum mechanics, rotation operators are used to describe rotations in the state space of a quantum system. 
# The rotation operators around the x, y, and z axes are denoted as Rx, Ry, and Rz, respectively. These operators are 
# represented by 2x2 unitary matrices. The general form of these matrices for a rotation by an angle θ are:
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
# These matrices are derived from the exponential of the Pauli matrices, which are the generators of rotations in quantum mechanics.


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
    
    # Ensure the matrix is unitary by rounding very small numerical errors towards zero
    R = np.where(np.abs(R) < 1e-10, 0, R)
    
    return R



# Background: In quantum computing, the Unitary Coupled Cluster (UCC) ansatz is a method used to approximate the wavefunction of a quantum system. 
# It is particularly useful in quantum chemistry for simulating molecular systems. The UCC ansatz involves applying a series of unitary transformations 
# to an initial state, often the Hartree-Fock state. In this problem, we are tasked with creating a trial wavefunction using the UCC ansatz with a 
# single parameter θ. The ansatz is given by the expression: 
# |\psi(\theta)\rangle = exp(-i * θ * Y1 * X2) |01\rangle, 
# where Y1 and X2 are Pauli-Y and Pauli-X operators acting on the first and second qubits, respectively. 
# The initial state |01\rangle is a two-qubit state where the first qubit is in state |0⟩ and the second qubit is in state |1⟩. 
# The task is to express this ansatz as a series of quantum gates acting on the initial state |00⟩.



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

    # Define the initial state |01⟩
    initial_state = np.array([0, 1, 0, 0]).reshape(4, 1)

    # Construct the operator Y1 * X2
    Y1_X2 = np.kron(Y, X)

    # Construct the unitary operator exp(-i * θ * Y1 * X2)
    U = expm(-1j * theta * Y1_X2)

    # Apply the unitary operator to the initial state |01⟩
    ansatz = U @ initial_state

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
