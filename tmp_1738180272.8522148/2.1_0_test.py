from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps



# Background: 
# The diffraction of light through a lens can be modeled using the principles of wave optics. 
# A Gaussian beam is a common model for laser beams, characterized by its bell-shaped intensity profile.
# When a Gaussian beam passes through a lens, it undergoes focusing, which can be described by the lens maker's formula and the wavefront transformation.
# The refractive index of the lens material affects how much the light bends when entering and exiting the lens.
# The curvature of the lens determines the focal length, which is crucial for focusing the beam.
# The intensity distribution of the light after passing through the lens can be computed using the Fresnel diffraction integral.
# Discretization parameters (mr2, ne2, mr0) are used to numerically evaluate the light field in a 2D space.



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

    # Constants for discretization
    mr2 = 51  # number of radial points in the simulation space
    ne2 = 61  # number of angular points around the circle
    mr0 = 81  # number of points along the radius of the initial light field distribution

    # Define the simulation grid
    r = np.linspace(0, R0, mr0)  # radial points
    theta = np.linspace(0, 2 * np.pi, ne2)  # angular points

    # Create a meshgrid for the simulation
    R, Theta = np.meshgrid(r, theta)

    # Calculate the initial Gaussian beam intensity distribution
    I0 = np.exp(-2 * (R / R0)**2)

    # Calculate the phase change due to the lens
    # Using the lens maker's formula: 1/f = (n-1) * (1/R1 - 1/R2)
    # For a symmetric lens, R1 = -R2 = RL, so 1/f = (n-1) * (2/RL)
    f = RL / (2 * (n - 1))
    phase_change = np.exp(-1j * (np.pi / (lambda_ * f)) * R**2)

    # Apply the phase change to the initial intensity distribution
    U = I0 * phase_change

    # Compute the intensity distribution on the observation plane
    # Using Fresnel diffraction integral (simplified for this context)
    Ie = np.abs(U)**2

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