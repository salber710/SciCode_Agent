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
    
    # Constants
    k = 2 * np.pi / l  # Wave number
    alpha = 4 * np.pi * epsilon_0 * a**3 * ((n**2 - 1) / (n**2 + 2))  # Polarizability using Clausius-Mossotti relation
    
    # Calculate the electric field amplitude from the power and beam waist
    E0_1 = np.sqrt(2 * P[0] / (c * epsilon_0 * np.pi * w**2))
    E0_2 = np.sqrt(2 * P[1] / (c * epsilon_0 * np.pi * w**2))
    
    # Dipole moments induced in each nanosphere
    p1 = alpha * E0_1
    p2 = alpha * E0_2
    
    # Distance vector and its magnitude
    R_vec = np.array([R, 0, 0])  # Considering R along the x-axis
    R_magnitude = np.linalg.norm(R_vec)
    
    # Unit vector in the direction of R
    R_hat = R_vec / R_magnitude
    
    # Interaction energy between the dipoles (assuming dipole-dipole interaction)
    U_dd = (1 / (4 * np.pi * epsilon_0 * R_magnitude**3)) * (
        np.dot(p1, p2) - 3 * np.dot(p1, R_hat) * np.dot(p2, R_hat)
    )
    
    # Optical binding force is given by the gradient of the interaction energy
    # Since we consider the distance along x-axis, the force is derivative of U_dd with respect to R
    F = -np.gradient(U_dd, R)
    
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