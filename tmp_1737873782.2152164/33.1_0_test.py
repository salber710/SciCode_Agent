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

    # Calculate the nearest-neighbor contributions
    d1 = np.array([a, 0])
    d2 = np.array([-a/2, sqrt(3)*a/2])
    d3 = np.array([-a/2, -sqrt(3)*a/2])

    # Calculate the phase factors for nearest-neighbor hopping
    nn_hopping = t1 * (np.exp(1j * np.dot([kx, ky], d1)) +
                       np.exp(1j * np.dot([kx, ky], d2)) +
                       np.exp(1j * np.dot([kx, ky], d3)))

    # Calculate the next-nearest-neighbor contributions
    d1_nn = np.array([3*a/2, sqrt(3)*a/2])
    d2_nn = np.array([-3*a/2, sqrt(3)*a/2])
    d3_nn = np.array([0, -sqrt(3)*a])

    # Calculate the phase factors for next-nearest-neighbor hopping
    nnn_hopping = t2 * (np.exp(1j * phi) * (np.exp(1j * np.dot([kx, ky], d1_nn)) +
                                            np.exp(1j * np.dot([kx, ky], d2_nn)) +
                                            np.exp(1j * np.dot([kx, ky], d3_nn))) +
                        np.exp(-1j * phi) * (np.exp(-1j * np.dot([kx, ky], d1_nn)) +
                                             np.exp(-1j * np.dot([kx, ky], d2_nn)) +
                                             np.exp(-1j * np.dot([kx, ky], d3_nn))))

    # Construct the Hamiltonian matrix
    hamiltonian = np.array([[m + nnn_hopping, nn_hopping],
                            [np.conj(nn_hopping), -m + nnn_hopping]])

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