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

    # Constants
    k = 2 * np.pi / lambda_  # Wavenumber

    # Discretization parameters
    mr2 = 51  # Number of radial points in the simulation space
    ne2 = 61  # Number of angular points around the circle
    mr0 = 81  # Number of points along the radius of the initial light field distribution
    
    # Define the simulation grid
    r = np.linspace(0, R0, mr0)  # Radial grid
    theta = np.linspace(0, 2 * np.pi, ne2)  # Angular grid

    # Calculate the field distribution
    R, T = np.meshgrid(r, theta)
    X = R * np.cos(T)
    Y = R * np.sin(T)

    # Calculate the input Gaussian beam profile
    w0 = R0 / np.sqrt(2)  # Beam waist
    E0 = np.exp(-(R**2) / (w0**2))  # Electric field amplitude

    # Propagation phase factor
    zR = np.pi * w0**2 / lambda_  # Rayleigh range
    q = zR * (1 - 1j * RL / zR)

    # Apply lens effect
    phi_lens = (k * n * d) * (1 - np.sqrt(1 - ((R**2) / (RL**2))))
    E_lens = E0 * np.exp(1j * phi_lens)

    # Calculate diffraction pattern at focus
    Ie = np.abs(E_lens)**2  # Intensity is the square of the amplitude

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