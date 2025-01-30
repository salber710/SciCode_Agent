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

    # Define the inverse lattice vectors in a different basis
    a1 = np.array([a, 0])
    a2 = np.array([-a/2, np.sqrt(3)*a/2])
    
    # Calculate exponential factors for nearest-neighbor terms
    exp_factors = np.array([
        np.exp(1j * (kx * a1[0] + ky * a1[1])),
        np.exp(1j * (kx * a2[0] + ky * a2[1])),
        np.exp(-1j * (kx * (a1[0] + a2[0]) + ky * (a1[1] + a2[1])))
    ])
    
    # Calculate the nearest-neighbor term
    f_k = t1 * np.sum(exp_factors)

    # Define phase factors for next-nearest-neighbor terms
    phase_factors = np.array([
        np.exp(1j * phi),
        np.exp(-1j * phi)
    ])
    
    # Calculate next-nearest-neighbor term in a different format
    g_k = t2 * (np.sum(phase_factors[0] * exp_factors) + np.sum(phase_factors[1] * np.conjugate(exp_factors)))

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