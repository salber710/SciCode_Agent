from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps



# Background: 
# The diffraction of light through a lens can be described using wave optics. A Gaussian beam, characterized by its
# intensity distribution, propagates in space and can be focused by a lens. The refractive index of the lens affects
# how light bends when passing through it, while the lens's curvature and thickness determine its focusing power.
# The interaction between the incident light and the lens can be modeled using the Huygens-Fresnel principle, which
# states that every point on a wavefront acts as a source of secondary spherical wavelets. The resulting wavefront is
# the superposition of these wavelets. In computational simulations, the beam and lens system is discretized over a
# grid. The intensity distribution of the light after passing through the lens can be analyzed on a specified plane
# by integrating the light field using numerical methods like Simpson's rule.



def simulate_light_diffraction(n, d, RL, R0, lambda_):
    '''Function to simulate light diffraction through a lens and plot the intensity distribution.
    Inputs:
    n (float): Refractive index of the lens material.
    d (float): Center thickness of the lens in mm.
    RL (float): Radius of curvature of the lens in mm.
    R0 (float): Radius of the incident light beam in mm.
    lambda_ (float): Wavelength of the incident light in mm.
    Outputs:
    Ie (numpy.ndarray): A 2D array of intensity values representing the diffraction pattern.
    '''
    
    # Constants for the simulation
    mr2 = 51  # Number of radial points for evaluation
    ne2 = 61  # Number of angular points
    mr0 = 81  # Number of points along the radius of the initial light field distribution
    
    # Discretization of the spatial domain
    r = np.linspace(0, R0, mr0)
    theta = np.linspace(0, 2 * np.pi, ne2)
    R, Theta = np.meshgrid(r, theta)
    
    # Initial Gaussian beam profile
    E0 = np.exp(-(R**2) / (R0**2))
    
    # Phase addition due to the lens
    # Assuming a thin lens approximation
    k = 2 * np.pi / lambda_  # Wave number
    phi_lens = -k * (n - 1) * (d - R**2 / (2 * RL))
    
    # Final field after lens
    E_final = E0 * np.exp(1j * phi_lens)
    
    # Compute the intensity distribution on the output plane
    # Perform integration over theta to get the intensity profile
    I = np.abs(E_final)**2
    Ie = simps(I, theta, axis=0)
    
    # Reshape the intensity distribution into a 2D array
    Ie = Ie.reshape((mr2, mr2))
    
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

