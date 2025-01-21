import numpy as np
import cmath
from math import pi, sin, cos, sqrt



# Background: The Haldane model describes electrons on a hexagonal lattice with both nearest-neighbor (NN) and next-nearest-neighbor (NNN) hopping. 
# It incorporates a complex phase in the NNN hopping term, which can lead to topologically nontrivial band structures. The Hamiltonian for the Haldane model is a 2x2 matrix.
# The NN hopping introduces off-diagonal terms in the Hamiltonian, while the NNN hopping contributes both to diagonal terms (with a phase) and an on-site energy term.
# The wavevector components (kx, ky) are part of the reciprocal space representation, and they affect the phase of the hopping terms due to the Bloch's theorem.
# The lattice structure of the hexagonal lattice contributes to the specific form of the Hamiltonian matrix.




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

    # Define the lattice vectors for the hexagonal lattice
    a1 = a * np.array([1, 0])
    a2 = a * np.array([-0.5, sqrt(3)/2])
    a3 = a * np.array([-0.5, -sqrt(3)/2])

    # Calculate the phase factors for nearest-neighbor hopping
    delta1 = np.array([1, 0])
    delta2 = np.array([-0.5, sqrt(3)/2])
    delta3 = np.array([-0.5, -sqrt(3)/2])

    # Calculate the nearest-neighbor contributions
    f_k = t1 * (np.exp(1j * (kx * delta1[0] + ky * delta1[1])) +
                np.exp(1j * (kx * delta2[0] + ky * delta2[1])) +
                np.exp(1j * (kx * delta3[0] + ky * delta3[1])))

    # Calculate next-nearest-neighbor contributions with phase
    g_k = 2 * t2 * (cos(kx * a1[0] + ky * a1[1] - phi) +
                    cos(kx * a2[0] + ky * a2[1] - phi) +
                    cos(kx * a3[0] + ky * a3[1] - phi))

    # Construct the Haldane Hamiltonian matrix
    hamiltonian = np.array([[m + g_k, f_k],
                            [np.conjugate(f_k), -(m + g_k)]])
    
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
