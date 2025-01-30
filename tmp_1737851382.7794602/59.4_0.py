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


# Background: In quantum mechanics, the expectation value of an observable is a key concept that provides the average 
# outcome of measurements of that observable on a quantum state. For a given quantum state |ψ⟩ and an observable 
# represented by an operator O, the expectation value is given by ⟨ψ|O|ψ⟩. In this problem, we are interested in 
# measuring the expectation value of the Pauli Z operator on the first qubit of a two-qubit system. The measurement 
# process involves applying a unitary transformation U to the state |ψ⟩, and then measuring the Z operator on the 
# first qubit. The Pauli Z operator for a single qubit is represented by the matrix Z = [[1, 0], [0, -1]]. For a 
# two-qubit system, measuring Z on the first qubit corresponds to the operator Z ⊗ I, where I is the identity matrix 
# for the second qubit. The expectation value is then calculated as ⟨ψ'|Z ⊗ I|ψ'⟩, where |ψ'⟩ = U|ψ⟩ is the state 
# after applying the unitary transformation.


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
    # Validate inputs
    if not isinstance(U, np.ndarray) or U.shape != (4, 4):
        raise ValueError("U must be a 4x4 matrix.")
    if not isinstance(psi, np.ndarray) or psi.shape != (4, 1):
        raise ValueError("psi must be a 4x1 vector.")
    if not np.allclose(U.conj().T @ U, np.eye(4), atol=1e-10):
        raise ValueError("U must be unitary.")
    if not np.allclose(np.vdot(psi, psi), 1, atol=1e-10):
        raise ValueError("psi must be normalized.")

    # Define the Pauli Z operator for a single qubit
    Z = np.array([[1, 0], [0, -1]])
    
    # Define the identity operator for a single qubit
    I = np.eye(2)
    
    # Construct the Z ⊗ I operator for the two-qubit system
    Z1_I = np.kron(Z, I)
    
    # Apply the unitary transformation U to the state psi
    psi_prime = U @ psi
    
    # Calculate the expectation value ⟨ψ'|Z ⊗ I|ψ'⟩
    measured_result = np.vdot(psi_prime, Z1_I @ psi_prime).real
    
    return measured_result



# Background: In quantum mechanics, the expectation value of an observable, such as the energy of a system, is calculated
# using the Hamiltonian operator. The Hamiltonian for a quantum system is a sum of terms, each representing a different
# interaction or energy contribution. In this problem, the Hamiltonian is given by:
# H = g0*I + g1*Z1 + g2*Z2 + g3*Z1Z2 + g4*Y1Y2 + g5*X1X2.
# To calculate the expectation value of the energy, we need to evaluate the expectation value of each term separately.
# This involves applying specific unitary transformations to the quantum state and measuring the resulting state.
# The expectation value for each term is calculated as ⟨ψ'|O|ψ'⟩, where O is the operator corresponding to the term,
# and |ψ'⟩ is the state after applying the unitary transformation. The total energy is the sum of these expectation values.



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
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    # Define the initial state |01⟩
    initial_state = np.array([0, 1, 0, 0]).reshape(4, 1)

    # Construct the operator Y1 * X2
    Y1_X2 = np.kron(Y, X)

    # Construct the unitary operator exp(-i * θ * Y1 * X2)
    U_ansatz = expm(-1j * theta * Y1_X2)

    # Apply the unitary operator to the initial state |01⟩
    psi = U_ansatz @ initial_state

    # Define the Hamiltonian terms
    Z1 = np.kron(Z, I)
    Z2 = np.kron(I, Z)
    Z1Z2 = np.kron(Z, Z)
    Y1Y2 = np.kron(Y, Y)
    X1X2 = np.kron(X, X)

    # Calculate expectation values for each term
    # g0 * I
    energy = gl[0]

    # g1 * Z1
    energy += gl[1] * measureZ(np.eye(4), psi)

    # g2 * Z2
    U_Z2 = np.kron(I, np.eye(2))
    energy += gl[2] * measureZ(U_Z2, psi)

    # g3 * Z1Z2
    U_Z1Z2 = np.eye(4)
    energy += gl[3] * measureZ(U_Z1Z2, psi)

    # g4 * Y1Y2
    U_Y1Y2 = expm(-1j * np.pi / 4 * (np.kron(Y, Y)))
    energy += gl[4] * measureZ(U_Y1Y2, psi)

    # g5 * X1X2
    U_X1X2 = expm(-1j * np.pi / 4 * (np.kron(X, X)))
    energy += gl[5] * measureZ(U_X1X2, psi)

    return energy

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
    Z = np.array([[1, 0], [0, -1]])
    
    # Define the identity operator for a single qubit
    I = np.eye(2)
    
    # Construct the Z ⊗ I operator for the two-qubit system
    Z1_I = np.kron(Z, I)
    
    # Apply the unitary transformation U to the state psi
    psi_prime = U @ psi
    
    # Calculate the expectation value ⟨ψ'|Z ⊗ I|ψ'⟩
    measured_result = np.vdot(psi_prime, Z1_I @ psi_prime).real
    
    return measured_result

from scicode.parse.parse import process_hdf5_to_tuple
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
