import numpy as np
import cmath
from math import pi, sin, cos, sqrt



# Background: The Haldane model is a theoretical model used to describe a two-dimensional electron system on a hexagonal lattice, such as graphene. 
# It includes both nearest-neighbor and next-nearest-neighbor interactions. The Hamiltonian for the Haldane model is a 2x2 matrix that describes 
# the energy of the system in terms of the wavevector components (kx, ky), lattice spacing (a), nearest-neighbor coupling (t1), 
# next-nearest-neighbor coupling (t2), a phase (phi) associated with the next-nearest-neighbor hopping, and an on-site energy (m).
# The nearest-neighbor hopping terms contribute to the off-diagonal elements of the Hamiltonian, while the next-nearest-neighbor terms 
# and the on-site energy contribute to the diagonal elements. The phase phi introduces a complex phase factor in the next-nearest-neighbor 
# hopping, which can lead to a non-trivial topological phase.




def calc_hamiltonian(kx, ky, a, t1, t2, phi, m):
    '''Function to generate the Haldane Hamiltonian with a given set of parameters.
    Inputs:
    kx : float
        The x component of the wavevector.
    ky : float
        The y component of the wavevector.
    a : float
        The lattice spacing, i.e., the length of one side of the hexagon.
    t1 : float
        The nearest-neighbor coupling constant.
    t2 : float
        The next-nearest-neighbor coupling constant.
    phi : float
        The phase ranging from -π to π.
    m : float
        The on-site energy.
    Output:
    hamiltonian : matrix of shape(2, 2)
        The Haldane Hamiltonian on a hexagonal lattice.
    '''

    # Define the nearest-neighbor vectors for a hexagonal lattice
    delta1 = np.array([1, 0]) * a
    delta2 = np.array([-0.5, sqrt(3)/2]) * a
    delta3 = np.array([-0.5, -sqrt(3)/2]) * a

    # Calculate the nearest-neighbor hopping terms
    gamma1 = t1 * (cmath.exp(1j * np.dot(delta1, [kx, ky])) +
                   cmath.exp(1j * np.dot(delta2, [kx, ky])) +
                   cmath.exp(1j * np.dot(delta3, [kx, ky])))

    # Define the next-nearest-neighbor vectors
    eta1 = np.array([1.5, sqrt(3)/2]) * a
    eta2 = np.array([0, -sqrt(3)]) * a
    eta3 = np.array([-1.5, sqrt(3)/2]) * a

    # Calculate the next-nearest-neighbor hopping terms
    gamma2 = t2 * (cmath.exp(1j * (np.dot(eta1, [kx, ky]) + phi)) +
                   cmath.exp(1j * (np.dot(eta2, [kx, ky]) + phi)) +
                   cmath.exp(1j * (np.dot(eta3, [kx, ky]) + phi)))

    gamma2_conj = t2 * (cmath.exp(-1j * (np.dot(eta1, [kx, ky]) + phi)) +
                        cmath.exp(-1j * (np.dot(eta2, [kx, ky]) + phi)) +
                        cmath.exp(-1j * (np.dot(eta3, [kx, ky]) + phi)))

    # Construct the Haldane Hamiltonian matrix
    hamiltonian = np.array([[m + gamma2, gamma1],
                            [np.conj(gamma1), -m + gamma2_conj]])

    return hamiltonian

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('33.1', 3)
target = targets[0]

kx = 1
ky = 1
a = 1
t1 = 1
t2 = 0.3
phi = 1
m = 1
assert np.allclose(calc_hamiltonian(kx, ky, a, t1, t2, phi, m), target)
target = targets[1]

kx = 0
ky = 1
a = 0.5
t1 = 1
t2 = 0.2
phi = 1
m = 1
assert np.allclose(calc_hamiltonian(kx, ky, a, t1, t2, phi, m), target)
target = targets[2]

kx = 1
ky = 0
a = 0.5
t1 = 1
t2 = 0.2
phi = 1
m = 1
assert np.allclose(calc_hamiltonian(kx, ky, a, t1, t2, phi, m), target)
