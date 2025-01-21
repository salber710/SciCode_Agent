import numpy as np
from scipy.integrate import simps



# Background: To simulate the diffraction of a Gaussian beam through a lens, we need to understand the optics involved:
# 1. A Gaussian beam has an intensity profile described by I(r) = I0 * exp(-2*(r/R0)^2), where I0 is the peak intensity 
#    and R0 is the beam radius.
# 2. When a light beam passes through a lens, it is focused according to the lens's focal properties, which are determined 
#    by its curvature, thickness, and the refractive index of the lens material.
# 3. The wavefront of the beam changes as it passes through the lens, and this can be described using Huygens' principle 
#    and Fresnel diffraction theory.
# 4. The resulting intensity distribution on a plane can be numerically computed using discretization techniques over radial 
#    and angular coordinates, as specified by mr2, ne2, and mr0.
# 5. The computation involves calculating the phase change introduced by the lens, applying the Fourier transform to switch 
#    from spatial to frequency domain (if needed), and then back to compute the real space intensity.



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
    k = 2 * np.pi / lambda_  # Wave number

    # Discretization parameters
    mr2 = 51  # number of radial points
    ne2 = 61  # number of angular points
    mr0 = 81  # number of points for the initial beam

    # Define the radial and angular grids
    r_space = np.linspace(0, R0, mr2)
    theta_space = np.linspace(0, 2 * np.pi, ne2)
    
    # Initial Gaussian beam profile
    I0 = 1  # Peak intensity. This can be normalized later.
    initial_intensity = I0 * np.exp(-2 * (r_space / R0) ** 2)

    # Calculate the phase change due to the lens
    # Assuming a symmetric lens, the optical path length difference is:
    # OPL = n*d - sqrt(RL^2 - r^2) + d, where the second term accounts for the lens curvature.
    phase_change = np.zeros(mr2)
    for i, r in enumerate(r_space):
        if r < RL:
            phase_change[i] = k * (n * d - np.sqrt(RL**2 - r**2) + d)

    # Initialize intensity distribution
    Ie = np.zeros((mr2, ne2))

    # Calculate the diffraction pattern
    for i, r in enumerate(r_space):
        for j, theta in enumerate(theta_space):
            # Calculate the complex amplitude including phase change
            amplitude = initial_intensity[i] * np.exp(1j * phase_change[i])
            # Convert to Cartesian coordinates for the intensity distribution
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Add to the intensity distribution (accumulating contributions)
            Ie[i, j] = np.abs(amplitude)**2

    # Normalize the intensity
    Ie /= np.max(Ie)

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
