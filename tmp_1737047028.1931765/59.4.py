import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm

# Background: In quantum mechanics, rotation operators are used to describe rotations in the state space of a quantum system. 
# The rotation operators around the x, y, and z axes are denoted as Rx, Ry, and Rz, respectively. These operators are 
# represented by 2x2 matrices and are parameterized by an angle θ. The matrices are defined as follows:
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
# These matrices are used to rotate quantum states around the respective axes by the angle θ.


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
    
    return R


# Background: In quantum computing, the Unitary Coupled Cluster (UCC) ansatz is a method used to approximate the wavefunction of a quantum system. 
# It is particularly useful in quantum chemistry for simulating molecular systems. The UCC ansatz involves applying a series of unitary transformations 
# to an initial state, often the Hartree-Fock state, to generate a trial wavefunction. In this problem, we are tasked with creating a trial wavefunction 
# using the UCC ansatz with a single parameter θ. The specific transformation involves the operator exp(-i θ Y1 X2), where Y1 and X2 are Pauli operators 
# acting on the first and second qubits, respectively. The initial state is |01⟩, which is a two-qubit state. The task is to express this transformation 
# as a series of quantum gates acting on the initial state |00⟩.



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

    # Create the initial state |00⟩
    initial_state = np.array([1, 0, 0, 0], dtype=complex).reshape(4, 1)

    # Create the |01⟩ state from |00⟩ using an X gate on the second qubit
    X2 = np.kron(I, X)
    state_01 = X2 @ initial_state

    # Construct the operator exp(-i θ Y1 X2)
    Y1 = np.kron(Y, I)
    X2 = np.kron(I, X)
    operator = expm(-1j * theta * Y1 @ X2)

    # Apply the operator to the |01⟩ state
    ansatz = operator @ state_01

    return ansatz


# Background: In quantum mechanics, the expectation value of an observable is a key concept that provides the average
# outcome of measurements of that observable on a quantum state. For a given quantum state |ψ⟩ and an observable 
# represented by an operator Ĥ, the expectation value is given by ⟨ψ|Ĥ|ψ⟩. In this problem, we are interested in 
# measuring the expectation value of the Pauli Z operator on the first qubit of a two-qubit system. The measurement 
# process involves applying a unitary transformation U to the state |ψ⟩, and then measuring the Z operator on the 
# first qubit. The Pauli Z operator for the first qubit in a two-qubit system is represented as Z1 ⊗ I, where Z1 is 
# the Pauli Z matrix for the first qubit and I is the identity matrix for the second qubit. The expectation value 
# after applying the unitary transformation U is calculated as ⟨ψ|U†(Z1 ⊗ I)U|ψ⟩, where U† is the conjugate transpose 
# of U.

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
    # Define the Pauli Z operator for the first qubit in a two-qubit system
    Z1 = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    Z1_I = np.kron(Z1, I)  # Z1 ⊗ I

    # Apply the unitary transformation U to the state |ψ⟩
    transformed_psi = U @ psi

    # Calculate the expectation value ⟨ψ|U†(Z1 ⊗ I)U|ψ⟩
    expectation_value = np.conj(transformed_psi.T) @ Z1_I @ transformed_psi

    # Since expectation_value is a 1x1 matrix, extract the scalar
    measured_result = expectation_value[0, 0].real

    return measured_result



# Background: In quantum mechanics, the expectation value of an observable, such as the energy of a system, is a key concept. 
# For a given Hamiltonian H and a quantum state |ψ⟩, the expectation value of the energy is given by ⟨ψ|H|ψ⟩. 
# In this problem, the Hamiltonian is given as H = g0*I + g1*Z1 + g2*Z2 + g3*Z1Z2 + g4*Y1Y2 + g5*X1X2, where I is the identity operator, 
# and Z1, Z2, Y1Y2, and X1X2 are Pauli operators acting on a two-qubit system. The expectation value of each term in the Hamiltonian 
# (except the constant term g0*I) is calculated separately using a unitary transformation followed by a measurement in the Z-basis. 
# The total expectation value of the energy is then the sum of these individual expectation values, weighted by their respective coefficients.



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

    # Create the initial state |00⟩
    initial_state = np.array([1, 0, 0, 0], dtype=complex).reshape(4, 1)

    # Create the |01⟩ state from |00⟩ using an X gate on the second qubit
    X2 = np.kron(I, X)
    state_01 = X2 @ initial_state

    # Construct the operator exp(-i θ Y1 X2)
    Y1 = np.kron(Y, I)
    X2 = np.kron(I, X)
    operator = expm(-1j * theta * Y1 @ X2)

    # Apply the operator to the |01⟩ state
    psi = operator @ state_01

    # Define the Pauli Z operator for the first qubit in a two-qubit system
    Z1 = np.kron(Z, I)
    Z2 = np.kron(I, Z)
    Z1Z2 = np.kron(Z, Z)
    Y1Y2 = np.kron(Y, Y)
    X1X2 = np.kron(X, X)

    # Calculate the expectation values for each term
    def measureZ(U, psi):
        transformed_psi = U @ psi
        expectation_value = np.conj(transformed_psi.T) @ Z1 @ transformed_psi
        return expectation_value[0, 0].real

    # Identity term
    energy = gl[0]

    # Z1 term
    energy += gl[1] * measureZ(np.eye(4), psi)

    # Z2 term
    U_Z2 = np.kron(I, np.eye(2))  # No transformation needed for Z2
    energy += gl[2] * measureZ(U_Z2, psi)

    # Z1Z2 term
    U_Z1Z2 = np.eye(4)  # No transformation needed for Z1Z2
    energy += gl[3] * measureZ(U_Z1Z2, psi)

    # Y1Y2 term
    U_Y1Y2 = expm(-1j * np.pi / 4 * X1X2)  # Transform to measure Y1Y2
    energy += gl[4] * measureZ(U_Y1Y2, psi)

    # X1X2 term
    U_X1X2 = expm(1j * np.pi / 4 * Y1Y2)  # Transform to measure X1X2
    energy += gl[5] * measureZ(U_X1X2, psi)

    return energy


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
