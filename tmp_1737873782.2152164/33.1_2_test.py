from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import cmath
from math import pi, sin, cos, sqrt



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
    a1 = np.array([sqrt(3)/2, 0.5]) * a
    a2 = np.array([0, 1]) * a
    a3 = np.array([-sqrt(3)/2, 0.5]) * a

    # Nearest-neighbor vectors
    nn_vectors = [a1, a2, a3]

    # Next-nearest-neighbor vectors
    nnn_vectors = [a1 + a2, a2 + a3, a3 + a1, -a1 - a2, -a2 - a3, -a3 - a1]

    # Calculate the nearest-neighbor hopping term
    h_nn = np.zeros((2, 2), dtype=complex)
    for delta in nn_vectors:
        phase_factor = cmath.exp(1j * (kx * delta[0] + ky * delta[1]))
        h_nn += t1 * phase_factor * np.array([[0, 1], [1, 0]])

    # Calculate the next-nearest-neighbor hopping term
    h_nnn = np.zeros((2, 2), dtype=complex)
    for delta in nnn_vectors:
        phase_factor = cmath.exp(1j * (kx * delta[0] + ky * delta[1]))
        direction = np.sign(delta[0] * delta[1])
        h_nnn += t2 * phase_factor * cmath.exp(1j * direction * phi) * np.array([[1, 0], [0, -1]])

    # On-site energy term
    h_on_site = m * np.array([[1, 0], [0, -1]])

    # Total Hamiltonian
    hamiltonian = h_nn + h_nnn + h_on_site

    return hamiltonian


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e