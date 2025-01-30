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



def measureZ(U, psi):
    # Apply the unitary transformation U to the state psi
    transformed_psi = np.dot(U, psi)
    
    # Compute the expectation value using the element-wise multiplication method
    # Extract the first and third elements of the transformed state
    first_element = transformed_psi[0]
    third_element = transformed_psi[2]
    
    # Compute the expectation value by taking the real part of the product of the conjugates
    expectation_value = (np.conj(first_element) * first_element - np.conj(third_element) * third_element).real
    
    return expectation_value



# Background: In quantum mechanics, the expectation value of an observable (such as energy) is calculated by taking the inner product of the state vector with the observable operator. For a Hamiltonian H, the expectation value is given by ⟨ψ|H|ψ⟩, where |ψ⟩ is the state vector. In this problem, the Hamiltonian is given as a linear combination of Pauli operators: H = g0*I + g1*Z1 + g2*Z2 + g3*Z1Z2 + g4*Y1Y2 + g5*X1X2. Each term in the Hamiltonian corresponds to a different measurement basis, which can be achieved by applying a specific unitary transformation to the state before measuring in the Z basis. The expectation value of each term is calculated separately and then combined using the coefficients g0, g1, ..., g5.



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
    # Define Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=np.complex_)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex_)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex_)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex_)

    # Create the initial state |01>
    initial_state = np.array([0, 1, 0, 0], dtype=np.complex_)

    # Create the ansatz wavefunction
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    R = np.kron(cos * I - 1j * sin * Y, cos * X - sin * X)
    psi = R @ initial_state

    # Define the unitary transformations for each term
    U_Z1 = np.kron(Z, I)
    U_Z2 = np.kron(I, Z)
    U_Z1Z2 = np.kron(Z, Z)
    U_Y1Y2 = np.kron(Y, Y)
    U_X1X2 = np.kron(X, X)

    # Calculate the expectation values for each term
    def measureZ(U, psi):
        transformed_psi = np.dot(U, psi)
        first_element = transformed_psi[0]
        third_element = transformed_psi[2]
        expectation_value = (np.conj(first_element) * first_element - np.conj(third_element) * third_element).real
        return expectation_value

    exp_Z1 = measureZ(U_Z1, psi)
    exp_Z2 = measureZ(U_Z2, psi)
    exp_Z1Z2 = measureZ(U_Z1Z2, psi)
    exp_Y1Y2 = measureZ(U_Y1Y2, psi)
    exp_X1X2 = measureZ(U_X1X2, psi)

    # Calculate the total energy
    energy = (gl[0] + 
              gl[1] * exp_Z1 + 
              gl[2] * exp_Z2 + 
              gl[3] * exp_Z1Z2 + 
              gl[4] * exp_Y1Y2 + 
              gl[5] * exp_X1X2)

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