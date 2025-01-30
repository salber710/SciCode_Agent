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

    # Define unit vectors in a different basis
    d1 = np.array([1, 0])
    d2 = np.array([-0.5, np.sqrt(3)/2])
    d3 = np.array([-0.5, -np.sqrt(3)/2])

    # Nearest-neighbor coupling using a different calculation method
    f_k = t1 * (np.exp(1j * (kx * d1[0] * a + ky * d1[1] * a)) +
                np.exp(1j * (kx * d2[0] * a + ky * d2[1] * a)) +
                np.exp(1j * (kx * d3[0] * a + ky * d3[1] * a)))

    # Next-nearest-neighbor vectors with a different phase arrangement
    nn_vectors = [d1 + d2, d2 + d3, d3 + d1, -d1 - d2, -d2 - d3, -d3 - d1]
    g_k = t2 * sum(np.exp(1j * (np.dot([kx, ky], nn) * a + phi * (-1)**i)) for i, nn in enumerate(nn_vectors))

    # Construct the Hamiltonian with a different matrix structure
    hamiltonian = np.array([[m + g_k, f_k],
                            [np.conj(f_k), -m + np.conj(g_k)]])

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