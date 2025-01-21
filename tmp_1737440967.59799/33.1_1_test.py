import numpy as np
import cmath
from math import pi, sin, cos, sqrt



# Background: The Haldane model is a tight-binding model to describe electrons in a two-dimensional honeycomb lattice, similar to graphene. 
# It includes both nearest-neighbor and next-nearest-neighbor interactions. The key feature of the Haldane model is the inclusion 
# of a complex phase in the next-nearest-neighbor hopping term, which can lead to a non-trivial topology and the quantum Hall effect 
# without an external magnetic field. The Hamiltonian for the Haldane model on a hexagonal lattice is a 2x2 matrix, given by:
# 
# H(k) = [[ m + 2t2*cos(phi)*sum_ni(cos(k·d_ni)) , t1*sum_ni(exp(-i*k·a_ni)) ],
#         [ t1*sum_ni(exp(i*k·a_ni)) , -(m + 2t2*cos(phi)*sum_ni(cos(k·d_ni))) ]]
#
# where a_ni are the vectors to the nearest neighbors, and d_ni are the vectors to the next-nearest neighbors. 
# The terms involving t2 and phi account for the complex next-nearest-neighbor hopping.




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

    # Nearest-neighbor vectors (a1, a2, a3)
    a1 = np.array([a * sqrt(3) / 2, a / 2])
    a2 = np.array([0, a])
    a3 = np.array([-a * sqrt(3) / 2, a / 2])

    # Next-nearest-neighbor vectors (d1, d2, d3)
    d1 = np.array([a * sqrt(3), 0])
    d2 = np.array([-a * sqrt(3) / 2, 3 * a / 2])
    d3 = np.array([-a * sqrt(3) / 2, -3 * a / 2])

    # Calculate the sum of the nearest-neighbor terms
    nn_sum = (
        cmath.exp(-1j * (kx * a1[0] + ky * a1[1])) +
        cmath.exp(-1j * (kx * a2[0] + ky * a2[1])) +
        cmath.exp(-1j * (kx * a3[0] + ky * a3[1]))
    )

    nn_sum_conj = np.conj(nn_sum)

    # Calculate the next-nearest-neighbor term
    nnn_sum = (
        cos(kx * d1[0] + ky * d1[1]) +
        cos(kx * d2[0] + ky * d2[1]) +
        cos(kx * d3[0] + ky * d3[1])
    )

    # Construct the Haldane Hamiltonian
    hamiltonian = np.array([
        [m + 2 * t2 * cos(phi) * nnn_sum, t1 * nn_sum],
        [t1 * nn_sum_conj, -(m + 2 * t2 * cos(phi) * nnn_sum)]
    ])

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
