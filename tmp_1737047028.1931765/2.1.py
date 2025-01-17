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

    # Constants and parameters
    k = 2 * np.pi / lambda_  # Wave number
    f = RL / (n - 1)  # Focal length of the lens using lens maker's formula

    # Discretization parameters
    mr2 = 51
    ne2 = 61
    mr0 = 81

    # Create a grid for the output intensity distribution
    r = np.linspace(-R0, R0, mr2)
    theta = np.linspace(0, 2 * np.pi, ne2)
    R, Theta = np.meshgrid(r, theta)

    # Calculate the initial Gaussian beam profile
    def gaussian_beam(r, R0):
        return np.exp(-r**2 / R0**2)

    # Calculate the phase change due to the lens
    def lens_phase(r, f):
        return np.exp(-1j * k * r**2 / (2 * f))

    # Calculate the field at the lens
    E0 = gaussian_beam(R, R0) * lens_phase(R, f)

    # Calculate the intensity distribution after the lens
    Ie = np.abs(E0)**2

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
