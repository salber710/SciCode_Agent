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

    # Initial state |01⟩
    initial_state = np.array([0, 1, 0, 0])  # |01⟩ in the computational basis

    # Pauli-Y and Pauli-X matrices for two qubits
    Y1 = np.array([[0, -1j], [1j, 0]])  # Y gate acting on the first qubit
    X2 = np.array([[0, 1], [1, 0]])     # X gate acting on the second qubit

    # Tensor product to represent Y1 ⊗ I and I ⊗ X2
    Y1_I = np.kron(Y1, np.eye(2))
    I_X2 = np.kron(np.eye(2), X2)

    # The operator -i * theta * Y1 * X2
    operator = -1j * theta * np.dot(Y1_I, I_X2)

    # Exponentiate the operator to get the unitary matrix
    U = expm(operator)

    # Apply the unitary to the initial state to get the ansatz wavefunction
    ansatz = np.dot(U, initial_state)

    return ansatz


def measureZ(U, psi):
    '''Perform a measurement in the Z-basis for a 2-qubit system where only Pauli Sz measurements are possible.
    The measurement is applied to the first qubit.
    Inputs:
    U : matrix of shape(4, 4)
        The unitary transformation to be applied before measurement.
    psi : array of shape (4, 1)
        The two-qubit state before the unitary transformation.
    Output:
    measured_result: float
        The result of the Sz measurement after applying U.
    '''

    # Apply the unitary transformation U to the state psi
    transformed_state = np.dot(U, psi)

    # Define the Z-basis measurement matrix for the first qubit (Z1 ⊗ I)
    Z1 = np.array([[1, 0], [0, -1]])  # Pauli-Z matrix for the first qubit
    I = np.eye(2)  # Identity matrix for the second qubit
    Z1_I = np.kron(Z1, I)  # Tensor product to represent Z1 ⊗ I

    # Calculate the expectation value of the Z1 measurement
    measured_result = np.vdot(transformed_state, np.dot(Z1_I, transformed_state)).real

    return measured_result



def projective_expected(theta, gl):
    '''Calculate the expectation value of the energy with proper unitary transformations.
    Input:
    theta : float
        The only variational parameter.
    gl = [g0, g1, g2, g3, g4, g5] : array in size 6
        Hamiltonian coefficients.
    Output:
    energy : float
        The expectation value of the energy with the given parameter theta.
    '''

    # Initial state |01⟩
    initial_state = np.array([0, 1, 0, 0])  # |01⟩ in the computational basis

    # Unitary transformations for each term in the Hamiltonian
    U_1 = np.eye(4)  # Identity for Z1
    U_2 = np.eye(4)  # Identity for Z2
    U_3 = np.eye(4)  # Identity for Z1Z2

    # For Y1Y2 and X1X2, use exp(-i * theta * Y1 * X2) and initial state |01⟩
    Y1 = np.array([[0, -1j], [1j, 0]])  # Y gate acting on the first qubit
    X2 = np.array([[0, 1], [1, 0]])     # X gate acting on the second qubit

    # Tensor product for two-qubit operators
    Y1_I = np.kron(Y1, np.eye(2))
    I_X2 = np.kron(np.eye(2), X2)

    # Unitary transformation for Y1Y2
    operator_Y1X2 = -1j * theta * np.dot(Y1_I, I_X2)
    U_4 = expm(operator_Y1X2)

    # Unitary transformation for X1X2
    X1 = np.array([[0, 1], [1, 0]])  # X gate acting on the first qubit
    I_X1 = np.kron(X1, np.eye(2))
    U_5 = expm(-1j * theta * np.dot(I_X1, I_X2))

    # Calculate expectation values
    def measure(U, psi):
        transformed_state = np.dot(U, psi)
        Z1 = np.array([[1, 0], [0, -1]])
        I = np.eye(2)
        Z1_I = np.kron(Z1, I)
        return np.vdot(transformed_state, np.dot(Z1_I, transformed_state)).real

    # Calculate the expectation values for each term
    E0 = gl[0]  # Constant term
    E1 = gl[1] * measure(U_1, initial_state)
    E2 = gl[2] * measure(U_2, initial_state)
    E3 = gl[3] * measure(U_3, initial_state)
    E4 = gl[4] * measure(U_4, initial_state)
    E5 = gl[5] * measure(U_5, initial_state)

    # Total energy expectation value
    energy = E0 + E1 + E2 + E3 + E4 + E5

    return energy


try:
    targets = process_hdf5_to_tuple('59.4', 3)
    target = targets[0]
    g0 = -0.4804
    g1 = -0.4347
    g2 = 0.3435
    g3 = 0.5716
    g4 = 0.0910
    g5 = 0.0910
    gl = [g0, g1, g2, g3, g4, g5]
    theta = 0
    assert np.allclose(projective_expected(theta, gl), target)

    target = targets[1]
    g0 = -0.4804
    g1 = -0.4347
    g2 = 0.3435
    g3 = 0.5716
    g4 = 0.0910
    g5 = 0.0910
    gl = [g0, g1, g2, g3, g4, g5]
    theta = np.pi / 6
    assert np.allclose(projective_expected(theta, gl), target)

    target = targets[2]
    g0 = -0.4804
    g1 = -0.4347
    g2 = 0.3435
    g3 = 0.5716
    g4 = 0.0910
    g5 = 0.0910
    gl = [g0, g1, g2, g3, g4, g5]
    theta = np.pi / 6
    assert np.allclose(projective_expected(theta, gl), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e