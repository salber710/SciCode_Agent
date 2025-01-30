from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps





def simulate_light_diffraction(n, d, RL, R0, lambda_):
    '''Function to simulate light diffraction through a lens and plot the intensity distribution.
    Inputs:
    n (float): Refractive index of the lens material, e.g., 1.5062 for K9 glass.
    d (float): Center thickness of the lens in millimeters (mm).
    RL (float): Radius of curvature of the lens in millimeters (mm), positive for convex surfaces.
    R0 (float): Radius of the incident light beam in millimeters (mm).
    lambda_ (float): Wavelength of the incident light in millimeters (mm), e.g., 1.064e-3 mm for infrared light.
    Outputs:
    Ie (numpy.ndarray): A 2D array of intensity values of the diffraction pattern where Ie[i][j] is the ith x and jth y.
    '''

    # Constants and parameters
    mr2 = 51
    ne2 = 61
    mr0 = 81
    pi = np.pi

    # Beam waist radius and focal parameter
    omega_0 = np.sqrt(lambda_ * RL / pi)
    f = pi * omega_0**2 / lambda_

    # Sampling space
    r = np.linspace(0, 5 * R0, mr2)
    theta = np.linspace(0, 2 * pi, ne2)
    r0 = np.linspace(0, R0, mr0)

    # Initialize intensity matrix
    Ie = np.zeros((mr2, ne2))

    # Calculate beam parameters
    b1 = 2 * pi * omega_0**2 / lambda_
    Z1 = 0  # Assuming beam waist is at the lens

    # Calculate new confocal parameters and waist after lens
    b2 = (b1 * f**2) / ((f - Z1)**2 + (b1 / 2)**2)
    omega_20 = np.sqrt(lambda_ * b2 / (2 * pi))

    # Calculate intensity distribution
    for i in range(mr2):
        for j in range(ne2):
            r_val = r[i]
            theta_val = theta[j]
            x = r_val * np.cos(theta_val)
            y = r_val * np.sin(theta_val)

            # Calculate terms for intensity
            Rz = Z1 * (1 + (pi * omega_0**2 / (lambda_ * Z1))**2)
            omega_z = omega_0 * np.sqrt(1 + (lambda_ * Z1 / (pi * omega_0**2))**2)

            # Electric field calculation
            E = (omega_0 / omega_z) * np.exp(- (x**2 + y**2) / omega_z**2) \
                * np.exp(-1j * (2 * pi / lambda_) * (Z1 + (x**2 + y**2) / (2 * Rz)))

            # Intensity calculation
            Ie[i, j] = np.abs(E)**2

    return Ie


try:
    targets = process_hdf5_to_tuple('2.1', 4)
    target = targets[0]
    n=1.5062
    d=3
    RL=0.025e3
    R0=1
    lambda_=1.064e-3
    assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)

    target = targets[1]
    n=1.5062
    d=4
    RL=0.05e3
    R0=1
    lambda_=1.064e-3
    assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)

    target = targets[2]
    n=1.5062
    d=2
    RL=0.05e3
    R0=1
    lambda_=1.064e-3
    assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)

    target = targets[3]
    n=1.5062
    d=2
    RL=0.05e3
    R0=1
    lambda_=1.064e-3
    intensity_matrix = simulate_light_diffraction(n, d, RL, R0, lambda_)
    max_index_flat = np.argmax(intensity_matrix)
    assert (max_index_flat==0) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e