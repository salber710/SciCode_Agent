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

    # Defining complex numbers for nearest-neighbor vectors
    z1 = np.exp(1j * kx * a)  # e^(i * kx * a)
    z2 = np.exp(1j * (kx * (-a/2) + ky * (np.sqrt(3) * a/2)))
    z3 = np.exp(1j * (kx * (-a/2) - ky * (np.sqrt(3) * a/2)))

    # Calculate nearest-neighbor hopping term
    f_k = t1 * (z1 + z2 + z3)

    # Defining phase factors for next-nearest-neighbor vectors
    e_iphi = np.exp(1j * phi)
    e_imphi = np.exp(-1j * phi)

    # Next-nearest-neighbor terms contributing to g_k
    g_k = t2 * (e_iphi * (z1 * z2 + z2 * z3 + z3 * z1) +
                e_imphi * (np.conjugate(z1) * np.conjugate(z2) +
                           np.conjugate(z2) * np.conjugate(z3) +
                           np.conjugate(z3) * np.conjugate(z1)))

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