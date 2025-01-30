from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import cmath
from math import pi, sin, cos, sqrt



# Background: The Haldane model is a key concept in condensed matter physics used to describe a Chern insulator on a hexagonal lattice. This model includes both nearest-neighbor and next-nearest-neighbor interactions, with the latter incorporating a complex phase that can lead to non-trivial topological properties. The Hamiltonian for the Haldane model on a hexagonal lattice is represented as a 2x2 matrix. The wavevector components (kx, ky) allow us to express the position in momentum space. The parameters include the lattice spacing (a), the coupling constants for nearest-neighbor (t1) and next-nearest-neighbor (t2) interactions, a phase (phi) for the next-nearest-neighbor hopping, and an on-site energy (m). The construction of the Hamiltonian involves complex exponentials due to the phase terms and the periodicity inherent in the lattice structure.




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

    # Constants for the hexagonal lattice structure
    sqrt3 = sqrt(3)

    # Nearest-neighbor vectors
    d1 = np.array([a, 0])
    d2 = np.array([-a/2, sqrt3*a/2])
    d3 = np.array([-a/2, -sqrt3*a/2])
    
    # Next-nearest-neighbor vectors
    nn1 = np.array([a, sqrt3*a])
    nn2 = np.array([-a, sqrt3*a])
    nn3 = np.array([-2*a, 0])
    nn4 = np.array([-a, -sqrt3*a])
    nn5 = np.array([a, -sqrt3*a])
    nn6 = np.array([2*a, 0])

    # Nearest-neighbor coupling term
    f_k = t1 * (np.exp(1j * np.dot([kx, ky], d1)) + np.exp(1j * np.dot([kx, ky], d2)) + np.exp(1j * np.dot([kx, ky], d3)))

    # Next-nearest-neighbor coupling term
    g_k = t2 * (np.exp(1j * phi) * (np.exp(1j * np.dot([kx, ky], nn1)) + np.exp(1j * np.dot([kx, ky], nn2)) + np.exp(1j * np.dot([kx, ky], nn3))) + 
                np.exp(-1j * phi) * (np.exp(1j * np.dot([kx, ky], nn4)) + np.exp(1j * np.dot([kx, ky], nn5)) + np.exp(1j * np.dot([kx, ky], nn6))))

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