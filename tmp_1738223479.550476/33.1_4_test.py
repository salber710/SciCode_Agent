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

    # Use a different approach by defining hexagonal lattice vectors in polar coordinates
    theta = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    nn_vectors = a * np.exp(1j * theta)
    nn_contributions = np.sum(np.exp(1j * (kx * np.real(nn_vectors) + ky * np.imag(nn_vectors))))

    # Calculate the nearest-neighbor hopping term
    f_k = t1 * nn_contributions

    # Calculate the next-nearest-neighbor hopping term using an alternative organization
    phi_vectors = np.array([np.pi / 3, np.pi, 5 * np.pi / 3])
    nnn_phase_factors = np.exp(1j * phi_vectors)
    nnn_vectors = np.array([
        a * (np.cos(phi_vectors) + 1j * np.sin(phi_vectors)),
        -a * (np.cos(phi_vectors) + 1j * np.sin(phi_vectors))
    ]).ravel()
    g_k = t2 * (np.sum(nnn_phase_factors * np.exp(1j * (kx * np.real(nnn_vectors) + ky * np.imag(nnn_vectors)))) +
                np.sum(np.conjugate(nnn_phase_factors) * np.exp(-1j * (kx * np.real(nnn_vectors) + ky * np.imag(nnn_vectors)))))

    # Construct the Haldane Hamiltonian matrix in a different format
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