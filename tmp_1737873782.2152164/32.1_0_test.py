from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy
from scipy.constants import epsilon_0, c





def binding_force(P, phi, R, l, w, a, n):
    '''Function to calculate the optical binding force between two trapped nanospheres.
    Input
    P : list of length 2
        Power of the two optical traps.
    phi : float
        Polarization direction of the optical traps.
    R : float
        Distance between the trapped nanospheres.
    l : float
        Wavelength of the optical traps.
    w : float
        Beam waist of the optical traps.
    a : float
        Radius of the trapped microspheres.
    n : float
        Refractive index of the trapped microspheres.
    Output
    F : float
        The optical binding force between two trapped nanospheres.
    '''
    # Calculate the wave number
    k = 2 * np.pi / l

    # Calculate the polarizability using the Clausius-Mossotti relation
    alpha = (3 * epsilon_0 * (n**2 - 1)) / (n**2 + 2) * (4/3) * np.pi * (a**3)

    # Calculate the electric field amplitude of the optical trap
    E0 = np.sqrt(2 * P[0] / (np.pi * w**2 * epsilon_0 * c))

    # Calculate the dipole moment induced in each nanosphere
    p1 = alpha * E0
    p2 = alpha * E0

    # Calculate the interaction force between the dipoles
    # Using the formula for dipole-dipole interaction
    F = (1 / (4 * np.pi * epsilon_0)) * (
        (3 * p1 * p2 * np.cos(phi) / R**4) - (p1 * p2 / R**3)
    )

    return F


try:
    targets = process_hdf5_to_tuple('32.1', 3)
    target = targets[0]
    P = [10000000, 100000000]
    phi = 0
    R = 1550e-9
    l = 1550e-9
    w = 600e-9
    a = 100e-9
    n = 1.444
    assert np.allclose(binding_force(P, phi, R, l, w, a, n), target)

    target = targets[1]
    P = [10000000, 100000000]
    phi = np.pi/2
    R = 1550e-9
    l = 1550e-9
    w = 600e-9
    a = 100e-9
    n = 1.444
    assert np.allclose(binding_force(P, phi, R, l, w, a, n), target)

    target = targets[2]
    P = [1000000000, 1000000000]
    phi = np.pi/4
    R = 1550e-9
    l = 1550e-9
    w = 600e-9
    a = 100e-9
    n = 1.444
    assert np.allclose(binding_force(P, phi, R, l, w, a, n), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e