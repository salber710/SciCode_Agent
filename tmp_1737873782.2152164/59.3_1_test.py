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
    
    # Define the Z operator for the first qubit: Z ⊗ I
    Z1 = np.array([[1, 0], [0, -1]])  # Pauli-Z matrix
    I = np.eye(2)  # Identity matrix
    Z1_I = np.kron(Z1, I)  # Z1 ⊗ I
    
    # Calculate the expectation value of Z1 ⊗ I
    measured_result = np.vdot(transformed_state, np.dot(Z1_I, transformed_state)).real
    
    return measured_result


try:
    targets = process_hdf5_to_tuple('59.3', 3)
    target = targets[0]
    CNOT21 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    U = CNOT21
    psi = np.kron([[0],[1]],[[0],[1]])
    assert np.allclose(measureZ(U, psi), target)

    target = targets[1]
    CNOT21 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    U = np.dot(CNOT21, np.kron(H, H))
    psi = np.kron([[1],[-1]],[[1],[-1]]) / 2
    assert np.allclose(measureZ(U, psi), target)

    target = targets[2]
    CNOT21 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    S = np.array([[1, 0], [0, 1j]])
    U = np.dot(CNOT21, np.kron(np.dot(H, S.conj().T), np.dot(H, S.conj().T)))
    psi = np.kron([[1],[-1j]],[[1],[-1j]]) / 2
    assert np.allclose(measureZ(U, psi), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e