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



def projective_expected(theta, gl):
    # Define Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Define the initial state |ψ⟩ = cos(θ/2)|01⟩ + sin(θ/2)|10⟩
    psi = np.array([0, np.cos(theta / 2), np.sin(theta / 2), 0], dtype=complex)

    # Measurement operators
    ZZ = np.kron(Z, Z)
    YY = np.kron(Y, Y)
    XX = np.kron(X, X)
    ZI = np.kron(Z, I)
    IZ = np.kron(I, Z)

    # Calculate expectation values using a dictionary to map operators to their coefficients
    operators = {'ZI': ZI, 'IZ': IZ, 'ZZ': ZZ, 'YY': YY, 'XX': XX}
    expectations = {key: np.vdot(psi, op @ psi).real for key, op in operators.items()}

    # Total energy calculation using dictionary comprehension
    energy = gl[0] + sum(gl[i + 1] * expectations[op] for i, op in enumerate(operators))

    return energy



# Background: The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm used to find the ground state energy of a quantum system. 
# It involves preparing a parameterized quantum state (ansatz) and measuring its energy with respect to a given Hamiltonian. 
# The parameters of the ansatz are optimized using a classical optimization algorithm to minimize the expectation value of the energy. 
# In this context, the parameter θ is varied to minimize the energy calculated using the Hamiltonian coefficients gl. 
# The optimization process involves evaluating the energy for different values of θ and finding the minimum value using a classical optimizer.

def perform_vqe(gl):
    '''Perform vqe optimization
    Input:
    gl = [g0, g1, g2, g3, g4, g5] : array in size 6
        Hamiltonian coefficients.
    Output:
    energy : float
        VQE energy.
    '''
    
    # Define the cost function to minimize
    def cost_function(theta):
        return projective_expected(theta, gl)
    
    # Use a classical optimizer to find the optimal theta that minimizes the energy
    result = minimize(cost_function, x0=0.0, method='BFGS')
    
    # The minimum energy found
    energy = result.fun
    
    return energy


try:
    targets = process_hdf5_to_tuple('59.5', 3)
    target = targets[0]
    def perform_diag(gl):
        """
        Calculate the ground-state energy with exact diagonalization
        Input:
        gl = [g0, g1, g2, g3, g4, g5] : array in size 6
            Hamiltonian coefficients.
        Output:
        energy : float
            The ground-state energy.
        """
        I = np.array([[1, 0], [0, 1]])
        Sx = np.array([[0, 1], [1, 0]])
        Sy = np.array([[0, -1j], [1j, 0]])
        Sz = np.array([[1, 0], [0, -1]])
        Ham = (gl[0] * np.kron(I, I) + # g0 * I
            gl[1] * np.kron(Sz, I) + # g1 * Z0
            gl[2] * np.kron(I, Sz) + # g2 * Z1
            gl[3] * np.kron(Sz, Sz) + # g3 * Z0Z1
            gl[4] * np.kron(Sy, Sy) + # g4 * Y0Y1
            gl[5] * np.kron(Sx, Sx))  # g5 * X0X1
        Ham_gs = np.linalg.eigvalsh(Ham)[0]# take the lowest value
        return Ham_gs
    g0 = -0.4804
    g1 = -0.4347
    g2 = 0.3435
    g3 = 0.5716
    g4 = 0.0910
    g5 = 0.0910
    gl = [g0, g1, g2, g3, g4, g5]
    assert (np.isclose(perform_diag(gl), perform_vqe(gl))) == target

    target = targets[1]
    def perform_diag(gl):
        """
        Calculate the ground-state energy with exact diagonalization
        Input:
        gl = [g0, g1, g2, g3, g4, g5] : array in size 6
            Hamiltonian coefficients.
        Output:
        energy : float
            The ground-state energy.
        """
        I = np.array([[1, 0], [0, 1]])
        Sx = np.array([[0, 1], [1, 0]])
        Sy = np.array([[0, -1j], [1j, 0]])
        Sz = np.array([[1, 0], [0, -1]])
        Ham = (gl[0] * np.kron(I, I) + # g0 * I
            gl[1] * np.kron(Sz, I) + # g1 * Z0
            gl[2] * np.kron(I, Sz) + # g2 * Z1
            gl[3] * np.kron(Sz, Sz) + # g3 * Z0Z1
            gl[4] * np.kron(Sy, Sy) + # g4 * Y0Y1
            gl[5] * np.kron(Sx, Sx))  # g5 * X0X1
        Ham_gs = np.linalg.eigvalsh(Ham)[0]# take the lowest value
        return Ham_gs
    g0 = -0.4989
    g1 = -0.3915
    g2 = 0.3288
    g3 = 0.5616
    g4 = 0.0925
    g5 = 0.0925
    gl = [g0, g1, g2, g3, g4, g5]
    assert (np.isclose(perform_diag(gl), perform_vqe(gl))) == target

    target = targets[2]
    def perform_diag(gl):
        """
        Calculate the ground-state energy with exact diagonalization
        Input:
        gl = [g0, g1, g2, g3, g4, g5] : array in size 6
            Hamiltonian coefficients.
        Output:
        energy : float
            The ground-state energy.
        """
        I = np.array([[1, 0], [0, 1]])
        Sx = np.array([[0, 1], [1, 0]])
        Sy = np.array([[0, -1j], [1j, 0]])
        Sz = np.array([[1, 0], [0, -1]])
        Ham = (gl[0] * np.kron(I, I) + # g0 * I
            gl[1] * np.kron(Sz, I) + # g1 * Z0
            gl[2] * np.kron(I, Sz) + # g2 * Z1
            gl[3] * np.kron(Sz, Sz) + # g3 * Z0Z1
            gl[4] * np.kron(Sy, Sy) + # g4 * Y0Y1
            gl[5] * np.kron(Sx, Sx))  # g5 * X0X1
        Ham_gs = np.linalg.eigvalsh(Ham)[0]# take the lowest value
        return Ham_gs
    g0 = -0.5463
    g1 = -0.2550
    g2 = 0.2779
    g3 = 0.5235
    g4 = 0.0986
    g5 = 0.0986
    gl = [g0, g1, g2, g3, g4, g5]
    assert (np.isclose(perform_diag(gl), perform_vqe(gl))) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e