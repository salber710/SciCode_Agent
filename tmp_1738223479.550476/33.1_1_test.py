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

    # Nearest-neighbor vectors (in complex plane representation)
    delta1 = np.exp(1j * (0))
    delta2 = np.exp(1j * (2 * np.pi / 3))
    delta3 = np.exp(1j * (4 * np.pi / 3))

    # Next-nearest-neighbor phase factors
    exp_iphi = np.exp(1j * phi)
    exp_imphi = np.exp(-1j * phi)

    # Nearest-neighbor coupling term
    f_k = t1 * (np.exp(1j * (kx * a + ky * 0)) * delta1 +
                np.exp(1j * (kx * -a/2 + ky * np.sqrt(3) * a/2)) * delta2 +
                np.exp(1j * (kx * -a/2 + ky * -np.sqrt(3) * a/2)) * delta3)

    # Next-nearest-neighbor coupling term
    g_k = t2 * (exp_iphi * (np.exp(1j * (kx * a + ky * np.sqrt(3) * a)) +
                            np.exp(1j * (kx * -a + ky * np.sqrt(3) * a)) +
                            np.exp(1j * (kx * -2 * a + ky * 0))) +
                exp_imphi * (np.exp(1j * (kx * a + ky * -np.sqrt(3) * a)) +
                             np.exp(1j * (kx * -a + ky * -np.sqrt(3) * a)) +
                             np.exp(1j * (kx * 2 * a + ky * 0))))

    # Construct the Haldane Hamiltonian matrix
    hamiltonian = np.array([[m + g_k, f_k],
                            [np.conj(f_k), -m + g_k]])

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