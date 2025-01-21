import numpy as np
from scipy.integrate import simps



# Background: To simulate the diffraction of a Gaussian light beam through a lens, we need to understand several optical principles. Gaussian beams are a common model for laser beams, characterized by their bell-shaped intensity profile. When such a beam passes through a lens, its phase is altered due to the refractive index difference, and it gets focused. The diffraction pattern can be calculated using the Huygens-Fresnel principle, which states that every point on a wavefront can be considered a source of secondary spherical wavelets. The resulting wave is the superposition of these wavelets. In this simulation, we'll discretize the beam and lens into a grid of points, compute the phase changes, and use numerical integration to find the resulting intensity distribution on a plane after the lens. The parameters mr2, ne2, and mr0 define the discretization resolution for our simulation.



def simulate_light_diffraction(n, d, RL, R0, lambda_):
    # Constants
    k = 2 * np.pi / lambda_  # Wave number
    
    # Define simulation grid
    r_vals = np.linspace(0, R0, mr0)  # Radial positions on the initial beam
    theta_vals = np.linspace(0, 2 * np.pi, ne2)  # Angular positions around the circle
    R, Theta = np.meshgrid(r_vals, theta_vals)  # Create a mesh for polar coordinates

    # Define the initial Gaussian beam profile
    E0 = np.exp(-(R**2) / (R0**2))

    # Phase change induced by the lens
    phase_lens = k * (d - (RL - np.sqrt(RL**2 - R**2))) * (n - 1)  # Lens phase shift
    E1 = E0 * np.exp(1j * phase_lens)

    # Prepare the grid for the output field after lens
    r_vals_out = np.linspace(0, R0, mr2)
    theta_vals_out = np.linspace(0, 2 * np.pi, ne2)
    R_out, Theta_out = np.meshgrid(r_vals_out, theta_vals_out)

    # Calculate the resulting field after the lens using numerical integration
    Ie = np.zeros((mr2, ne2), dtype=np.float64)

    for i in range(mr2):
        for j in range(ne2):
            # Compute the integral over the initial field
            r_diff = R_out[i, j] - R
            theta_diff = Theta_out[i, j] - Theta
            phase_diff = k * r_diff  # Simplification for small angles
            integrand = E1 * np.exp(1j * phase_diff)
            # Perform integration over r and theta
            integral_result = simps(simps(integrand, r_vals, axis=0), theta_vals)
            Ie[i, j] = np.abs(integral_result)**2  # Intensity is the square of the amplitude

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
