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
    c = 3e8  # Speed of light in m/s
    k = 2 * np.pi / lambda_  # Wave number

    # Discretization parameters
    mr2 = 51  # Number of radial points in the discretized simulation space
    ne2 = 61  # Number of angular points around the circle
    mr0 = 81  # Number of points along the radius of the initial light field distribution

    # Calculate the focal length of the lens using the lensmaker's formula
    f = RL / ((n - 1) * (1 - (d / RL) * ((n - 1) / n)))

    # Define spatial grid for simulation
    r = np.linspace(0, R0, mr2)  # Radial points
    theta = np.linspace(0, 2 * np.pi, ne2)  # Angular points

    # Create mesh grid for radial and angular coordinates
    R, Theta = np.meshgrid(r, theta)

    # Calculate the Gaussian beam profile before the lens
    w0 = R0
    I0 = np.exp(-2 * (R / w0) ** 2)

    # Propagate the Gaussian beam through the lens
    # Apply lens phase shift
    phase_shift = np.exp(-1j * k * R**2 / (2 * f))
    U = I0 * phase_shift

    # Calculate the intensity of the beam on the focal plane
    # The intensity is proportional to the square of the absolute value of the field
    Ie = np.abs(U)**2

    # Integrate over the angular coordinate to get the radial intensity distribution
    radial_intensity = simps(Ie, theta, axis=0)

    # Create a 2D intensity grid
    Ie_2D = np.zeros((mr2, mr2))
    for i in range(mr2):
        for j in range(mr2):
            r_i = np.sqrt(r[i]**2 + r[j]**2)
            if r_i < R0:
                Ie_2D[i, j] = radial_intensity[np.argmin(np.abs(r - r_i))]

    return Ie_2D


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