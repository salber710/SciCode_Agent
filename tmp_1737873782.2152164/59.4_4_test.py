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

    # Initialize the energy with the constant term
    energy = gl[0]

    # Create the ansatz wavefunction
    psi = create_ansatz(theta)

    # Define the unitary transformations for each term in the Hamiltonian
    U1 = np.eye(4)  # Identity for Z1 term
    U2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])  # Swap for Z2 term
    U3 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])  # CNOT for Z1Z2 term
    U4 = np.array([[0, 0, 0, -1j], [0, 0, -1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]])  # Y1Y2 term
    U5 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])  # X1X2 term

    # Calculate the expectation values for each term
    energy += gl[1] * measureZ(U1, psi)
    energy += gl[2] * measureZ(U2, psi)
    energy += gl[3] * measureZ(U3, psi)
    energy += gl[4] * measureZ(U4, psi)
    energy += gl[5] * measureZ(U5, psi)

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