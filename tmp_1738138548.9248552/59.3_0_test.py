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



def create_ansatz(theta):
    '''Create the ansatz wavefunction with a given theta.
    Input:
    theta : float
        The only variational parameter.
    Output:
    ansatz : array of shape (4, 1)
        The ansatz wavefunction.
    '''
    # Define the identity and Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=np.complex_)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex_)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex_)
    
    # Define the initial state |01>
    initial_state = np.array([0, 1, 0, 0], dtype=np.complex_)

    # Define the rotation matrix R = exp(-i * theta * Y1 * X2)
    # Using the formula for the exponential of a Kronecker product of Pauli matrices
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    R = np.kron(cos * I - 1j * sin * Y, cos * X - sin * X)

    # Apply the rotation matrix R to the initial state
    ansatz = R @ initial_state

    return ansatz.reshape(4, 1)



# Background: In quantum mechanics, the expectation value of an observable is a key concept that provides the average
# outcome of measurements of that observable on a quantum state. For a given quantum state |ψ⟩ and an observable 
# represented by an operator O, the expectation value is given by ⟨ψ|O|ψ⟩. In this problem, we are interested in 
# measuring the expectation value of the Pauli Z operator on the first qubit of a two-qubit system. The Pauli Z 
# operator for a single qubit is represented by the matrix Z = [[1, 0], [0, -1]]. For a two-qubit system, measuring 
# Z on the first qubit corresponds to the operator Z ⊗ I, where I is the identity matrix for the second qubit. 
# The process involves applying a unitary transformation U to the state |ψ⟩, and then computing the expectation 
# value of Z ⊗ I on the transformed state.

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
    # Define the Pauli Z operator for a single qubit
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex_)
    
    # Define the identity operator for a single qubit
    I = np.array([[1, 0], [0, 1]], dtype=np.complex_)
    
    # Construct the Z ⊗ I operator for the two-qubit system
    Z1 = np.kron(Z, I)
    
    # Apply the unitary transformation U to the state psi
    transformed_psi = U @ psi
    
    # Calculate the expectation value of Z1 on the transformed state
    measured_result = np.vdot(transformed_psi, Z1 @ transformed_psi).real
    
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