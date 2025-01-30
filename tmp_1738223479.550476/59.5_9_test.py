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
    # Create rotation matrices using numpy's trigonometric functions and numpy's complex numbers
    if axis == 1:  # R_x
        R = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], 
                      [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
    elif axis == 2:  # R_y
        R = np.array([[np.cos(theta/2), -np.sin(theta/2)], 
                      [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
    elif axis == 3:  # R_z
        R = np.array([[np.exp(-1j*theta/2), 0], 
                      [0, np.exp(1j*theta/2)]], dtype=np.complex128)
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
    # Define the initial state |00⟩
    initial_state = np.array([[1], [0], [0], [0]], dtype=np.complex128)

    # Prepare the |01⟩ state by applying an X gate on the second qubit
    X = np.array([[0, 1],
                  [1, 0]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    X_2 = np.kron(I, X)
    state_01 = np.dot(X_2, initial_state)

    # Define the Pauli-Y matrix
    Y = np.array([[0, -1j],
                  [1j, 0]], dtype=np.complex128)

    # Construct the Y_1 X_2 operator using Kronecker products
    Y_1 = np.kron(Y, I)
    Y1_X2 = np.dot(Y_1, X_2)

    # Use the Lie group approach to construct the unitary operator
    # For a skew-Hermitian matrix A, exp(-i theta A) can be written as
    # U = exp(-i theta Y1_X2) using the matrix exponential definition
    # U = I + A + A^2/2! + A^3/3! + ...
    I_4 = np.eye(4, dtype=np.complex128)
    A = -1j * theta * Y1_X2
    A2 = np.dot(A, A)
    A3 = np.dot(A2, A)
    U = I_4 + A + A2 / 2. + A3 / 6.

    # Apply the unitary operator to the |01⟩ state
    ansatz = np.dot(U, state_01)

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


    # Define the Pauli Z operator for the first qubit in a two-qubit system: Z ⊗ I
    Z1 = np.array([[1, 0, 0, 0], 
                   [0, 1, 0, 0], 
                   [0, 0, -1, 0], 
                   [0, 0, 0, -1]], dtype=np.complex128)

    # Transform the state psi using the unitary U
    psi_prime = np.matmul(U, psi)

    # Compute the expectation value using a spectral decomposition approach
    # Find eigenvalues and eigenvectors of Z1
    eigenvalues, eigenvectors = np.linalg.eigh(Z1)

    # Project psi_prime onto the eigenvectors of Z1
    projections = np.matmul(eigenvectors.conj().T, psi_prime)

    # Calculate the expectation value by summing weighted contributions of eigenvalues
    measured_result = sum(eigenvalues[i] * np.abs(projections[i])**2 for i in range(len(eigenvalues)))

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

    # Define the Pauli matrices
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Initial state |00⟩
    initial_state = np.array([1, 0, 0, 0], dtype=np.complex128)

    # Create the ansatz state |ψ(θ)⟩ using a combination of X and Y rotations
    def create_ansatz(theta):
        R1 = np.kron(expm(-1j * theta / 3 * X), I)
        R2 = np.kron(I, expm(-1j * theta / 3 * Y))
        R3 = np.kron(expm(-1j * theta / 3 * Z), expm(-1j * theta / 3 * Z))
        ansatz = np.dot(R3, np.dot(R2, np.dot(R1, initial_state)))
        return ansatz

    # Prepare the ansatz
    psi = create_ansatz(theta)

    # Function to measure expectation value ⟨ψ|O|ψ⟩
    def measure_op(O, psi):
        return np.vdot(psi, np.dot(O, psi)).real

    # Terms in the Hamiltonian
    g0, g1, g2, g3, g4, g5 = gl

    # Define Pauli operators for two qubits
    Z1 = np.kron(Z, I)
    Z2 = np.kron(I, Z)
    Z1Z2 = np.kron(Z, Z)
    Y1Y2 = np.kron(Y, Y)
    X1X2 = np.kron(X, X)

    # 1. Measure ⟨Z1⟩
    E_Z1 = measure_op(Z1, psi)

    # 2. Measure ⟨Z2⟩
    E_Z2 = measure_op(Z2, psi)

    # 3. Measure ⟨Z1Z2⟩
    E_Z1Z2 = measure_op(Z1Z2, psi)

    # 4. Measure ⟨Y1Y2⟩ using a transformation with a mix of X and Y
    U_Y1Y2 = np.kron(expm(-1j * np.pi / 3 * X), expm(-1j * np.pi / 3 * Y))
    psi_Y1Y2 = np.dot(U_Y1Y2, psi)
    E_Y1Y2 = measure_op(Z1Z2, psi_Y1Y2)

    # 5. Measure ⟨X1X2⟩ using a transformation with a mix of Y and Z
    U_X1X2 = np.kron(expm(-1j * np.pi / 3 * Y), expm(-1j * np.pi / 3 * Z))
    psi_X1X2 = np.dot(U_X1X2, psi)
    E_X1X2 = measure_op(Z1Z2, psi_X1X2)

    # Compute the energy expectation value
    energy = g0 + g1 * E_Z1 + g2 * E_Z2 + g3 * E_Z1Z2 + g4 * E_Y1Y2 + g5 * E_X1X2

    return energy





def perform_vqe(gl):
    '''Perform vqe optimization
    Input:
    gl = [g0, g1, g2, g3, g4, g5] : array in size 6
        Hamiltonian coefficients.
    Output:
    energy : float
        VQE energy.
    '''

    # Define a cost function using a unique quantum state representation
    def cost_function(theta):
        # Use a Chebyshev polynomial-based ansatz
        psi = np.array([np.cos(theta) + theta**3, np.sin(theta) + theta**2])
        # Construct a Hamiltonian matrix emphasizing diagonal elements
        H = np.array([[gl[0], gl[4]], [gl[4], gl[3]]])
        energy = np.real(np.vdot(psi, H @ psi))
        return energy

    # Use Conjugate Gradient method for optimization
    initial_theta = np.pi / 2  # A central initial guess
    result = cg(lambda x: np.array([cost_function(x)]), initial_theta, tol=1e-6)

    # The minimum energy found by the optimizer
    energy = cost_function(result[0])

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