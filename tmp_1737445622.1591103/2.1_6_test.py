from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps



# Background: 
# Diffraction through a lens for a Gaussian beam involves understanding how a lens focuses light
# and how the phase change induced by the lens affects the light's propagation. A Gaussian beam 
# is characterized by its waist (the smallest spot size) and spreads out as it propagates. 
# When this beam passes through a lens, the lens imparts a phase shift that can be expressed as 
# a quadratic function of the radial distance from the optical axis. This phase shift focuses the beam.
# The intensity distribution on the focal plane can be obtained by computing the Fourier transform 
# of the field at the lens, considering the lens's phase transformation and the diffraction effects.
# We discretize the space into radial and angular components to compute this distribution numerically.



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
    k = 2 * np.pi / lambda_  # Wavenumber

    # Discretization parameters
    mr2 = 51  # Number of radial points in the output
    ne2 = 61  # Number of angular points

    # Create radial and angular grids
    r = np.linspace(0, 2 * R0, mr2)  # Radial grid
    theta = np.linspace(0, 2 * np.pi, ne2)  # Angular grid

    # Compute the lens phase shift
    # Lensmaker's equation: phase shift due to the lens
    def lens_phase_shift(r):
        return -(k / (2 * RL)) * (r**2)

    # Initial Gaussian light field
    def gaussian_beam(r):
        return np.exp(-(r**2) / (R0**2))

    # Compute the field after the lens
    field_after_lens = np.zeros((mr2, ne2), dtype=complex)
    for i in range(mr2):
        for j in range(ne2):
            r_val = r[i]
            theta_val = theta[j]
            # Apply the Gaussian beam profile and the lens phase shift
            E0 = gaussian_beam(r_val) * np.exp(1j * lens_phase_shift(r_val))
            field_after_lens[i, j] = E0
    
    # Compute the intensity distribution
    # Here we integrate over the angular component using Simpson's rule
    Ie = np.zeros((mr2, mr2))
    for i in range(mr2):
        for j in range(mr2):
            # Convert (i, j) to polar coordinates
            r_i = r[i]
            r_j = r[j]
            integrand = field_after_lens * np.exp(1j * k * (r_i * np.cos(theta) + r_j * np.sin(theta)))
            Ie[i, j] = simps(np.abs(integrand)**2, theta)

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

