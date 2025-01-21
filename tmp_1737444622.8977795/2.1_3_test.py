import numpy as np
from scipy.integrate import simps



# Background: 
# The problem involves simulating the diffraction of a Gaussian beam as it passes through a lens. 
# The lens focuses the light, creating an intensity distribution on a plane. 
# The diffraction pattern can be described using Gaussian optics and the Fresnel diffraction integral.
# The Gaussian beam is characterized by its beam waist (radius R0), and its field can be described by the Gaussian function.
# The lens affects the phase of the light passing through it, which can be modeled using its refractive index, thickness, and curvature.
# The Fresnel approximation is used to compute the diffraction pattern at a given distance from the lens.
# The discretized simulation uses radial points (mr2), angular points (ne2), and initial radial points (mr0) to evaluate the field.



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

    # Create a grid for the initial field
    r0 = np.linspace(0, R0, mr0)
    theta0 = np.linspace(0, 2 * np.pi, ne2)
    r0_grid, theta0_grid = np.meshgrid(r0, theta0, indexing='ij')

    # Gaussian beam field at the lens
    E0 = np.exp(-r0_grid**2 / R0**2)

    # Lens phase transformation
    lens_phase = np.exp(-1j * (k * d - k * (n - 1) * (RL - np.sqrt(RL**2 - r0_grid**2))))

    # Field after passing through the lens
    E_lens = E0 * lens_phase

    # Create a grid for the output field
    r = np.linspace(0, R0, mr2)
    theta = np.linspace(0, 2 * np.pi, ne2)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')

    # Initialize the output field
    E_out = np.zeros_like(r_grid, dtype=complex)

    # Perform the integration using Fresnel approximation
    for i in range(mr2):
        for j in range(ne2):
            r_i = r[i]
            theta_j = theta[j]
            integrand = E_lens * np.exp(1j * k * (r_i**2 + r0_grid**2 - 2 * r_i * r0_grid * np.cos(theta_j - theta0_grid)) / (2 * RL))
            E_out[i, j] = simps(simps(integrand, theta0, axis=1), r0)

    # Intensity distribution
    Ie = np.abs(E_out)**2

    return Ie

from scicode.parse.parse import process_hdf5_to_tuple
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
