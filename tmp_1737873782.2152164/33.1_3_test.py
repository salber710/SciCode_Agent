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
    
    # Calculate the nearest-neighbor terms
    d1 = a * np.array([1, 0])
    d2 = a * np.array([-0.5, sqrt(3)/2])
    d3 = a * np.array([-0.5, -sqrt(3)/2])
    
    # Real space vectors
    delta_k_dot_d1 = kx * d1[0] + ky * d1[1]
    delta_k_dot_d2 = kx * d2[0] + ky * d2[1]
    delta_k_dot_d3 = kx * d3[0] + ky * d3[1]
    
    # Nearest-neighbor Hamiltonian terms
    h1 = t1 * (np.exp(1j * delta_k_dot_d1) + np.exp(1j * delta_k_dot_d2) + np.exp(1j * delta_k_dot_d3))
    
    # Calculate the next-nearest-neighbor terms with phase
    n1 = a * np.array([3/2, sqrt(3)/2])
    n2 = a * np.array([-3/2, sqrt(3)/2])
    n3 = a * np.array([0, -sqrt(3)])
    
    # Phase factors
    phase_factor1 = np.exp(1j * phi)
    phase_factor2 = np.exp(-1j * phi)
    
    # Next-nearest-neighbor Hamiltonian terms
    h2 = t2 * (phase_factor1 * np.exp(1j * (kx * n1[0] + ky * n1[1])) +
               phase_factor2 * np.exp(1j * (kx * n2[0] + ky * n2[1])) +
               phase_factor1 * np.exp(1j * (kx * n3[0] + ky * n3[1])))
    
    # On-site energy
    m_term = m
    
    # Construct the 2x2 Hamiltonian matrix
    hamiltonian = np.array([[m_term + h2, h1],
                            [np.conj(h1), -m_term + h2]])
    
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