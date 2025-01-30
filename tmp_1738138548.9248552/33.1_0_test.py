from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import cmath
from math import pi, sin, cos, sqrt



# Background: The Haldane model is a theoretical model used to describe a two-dimensional electron system on a hexagonal lattice, such as graphene. 
# It includes both nearest-neighbor and next-nearest-neighbor interactions. The Hamiltonian for the Haldane model is a 2x2 matrix that describes 
# the energy of the system in terms of the wavevector components (kx, ky), lattice spacing (a), nearest-neighbor coupling (t1), 
# next-nearest-neighbor coupling (t2), a phase (phi) associated with the next-nearest-neighbor hopping, and an on-site energy (m).
# The nearest-neighbor interactions are typically represented by a hopping term t1, while the next-nearest-neighbor interactions include 
# a complex phase factor e^(i*phi) or e^(-i*phi) depending on the direction of hopping. The on-site energy m introduces a mass term that can 
# open a gap in the energy spectrum. The Hamiltonian matrix is constructed using these parameters and the wavevector components.




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
    a1 = np.array([sqrt(3)/2, 1/2]) * a
    a2 = np.array([sqrt(3)/2, -1/2]) * a
    a3 = np.array([0, -1]) * a

    # Nearest-neighbor vectors
    delta1 = a1
    delta2 = a2
    delta3 = a3

    # Next-nearest-neighbor vectors
    b1 = a1 - a2
    b2 = a2 - a3
    b3 = a3 - a1

    # Calculate the nearest-neighbor hopping term
    f_k = t1 * (np.exp(1j * np.dot([kx, ky], delta1)) +
                np.exp(1j * np.dot([kx, ky], delta2)) +
                np.exp(1j * np.dot([kx, ky], delta3)))

    # Calculate the next-nearest-neighbor hopping term
    g_k = t2 * (np.exp(1j * (np.dot([kx, ky], b1) + phi)) +
                np.exp(1j * (np.dot([kx, ky], b2) + phi)) +
                np.exp(1j * (np.dot([kx, ky], b3) + phi)))

    # Construct the Haldane Hamiltonian
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