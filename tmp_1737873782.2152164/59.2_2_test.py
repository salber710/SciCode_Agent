from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm


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
    if axis == 1:
        # Rotation around x-axis
        R = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                      [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    elif axis == 2:
        # Rotation around y-axis
        R = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                      [np.sin(theta / 2), np.cos(theta / 2)]])
    elif axis == 3:
        # Rotation around z-axis
        R = np.array([[exp(-1j * theta / 2), 0],
                      [0, exp(1j * theta / 2)]])
    else:
        raise ValueError("Invalid axis. Axis must be 1 (x), 2 (y), or 3 (z).")

    return R





def create_ansatz(theta):
    '''Create the ansatz wavefunction with a given theta.
    Input:
    theta : float
        The only variational parameter.
    Output:
    ansatz : array of shape (4, 1)
        The ansatz wavefunction.
    '''

    # Initial state |01⟩ in the computational basis
    initial_state = np.array([0, 1, 0, 0], dtype=complex).reshape(4, 1)

    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    # Identity matrix
    I = np.eye(2, dtype=complex)

    # Construct the operator -iθY1X2, which acts on two qubits
    operator = -1j * theta * np.kron(Y, X)

    # The unitary transformation exp(-iθY1X2)
    U = expm(operator)

    # Apply the unitary transformation to the initial state
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