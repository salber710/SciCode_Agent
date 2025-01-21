import numpy as np
from scipy.integrate import simps



# Background: The task involves simulating the diffraction of a Gaussian beam through a lens and computing the intensity distribution on a plane. The Gaussian beam is characterized by its waist (radius) and wavelength. When a light beam passes through a lens, it undergoes phase changes due to the lens's refractive index and curvature. The lens focuses the beam, altering its intensity distribution. The simulation requires discretization of the space into radial and angular components to compute the electric field. The intensity distribution is derived from the electric field's magnitude. The Fresnel diffraction integral or similar approaches can be employed to model the propagation of the light field. The parameters mr2, ne2, and mr0 are used to discretize the simulation space.



def simulate_light_diffraction(n, d, RL, R0, lambda_):
    '''Function to simulate light diffraction through a lens and plot the intensity distribution.
    Inputs:
    n (float): Refractive index of the lens material.
    d (float): Center thickness of the lens in millimeters (mm).
    RL (float): Radius of curvature of the lens in millimeters (mm).
    R0 (float): Radius of the incident light beam in millimeters (mm).
    lambda_ (float): Wavelength of the incident light in millimeters (mm).
    Outputs:
    Ie (numpy.ndarray): A 2D array of intensity values of the diffraction pattern.
    '''

    # Discretization parameters
    mr2 = 51  # Number of radial points in the simulation space
    ne2 = 61  # Number of angular points around the circle
    mr0 = 81  # Number of points along the radius of the initial light field

    # Physical constants and pre-calculations
    k = 2 * np.pi / lambda_  # Wave number
    f = RL / (n - 1)  # Focal length of the lens by lensmaker's formula

    # Create coordinate grids for calculation
    r = np.linspace(0, R0, mr0)  # Radial points for initial field
    theta = np.linspace(0, 2 * np.pi, ne2)  # Angular points
    R, Theta = np.meshgrid(r, theta)

    # Initial Gaussian beam field
    E0 = np.exp(-(R**2) / (R0**2))

    # Phase change introduced by the lens
    phi_lens = (k / (2 * f)) * R**2

    # Electric field after lens
    E1 = E0 * np.exp(1j * phi_lens)

    # Creating grid for the output plane
    r2 = np.linspace(0, 2 * R0, mr2)
    theta2 = np.linspace(0, 2 * np.pi, ne2)
    R2, Theta2 = np.meshgrid(r2, theta2)

    # Simulate field at observation plane (using a simple Fourier optics approach)
    # For simplicity, assume the observation plane is at the focal plane
    E2 = np.zeros((mr2, ne2), dtype=complex)

    for i, r_val in enumerate(r2):
        for j, theta_val in enumerate(theta2):
            # Integrating over initial field distribution
            integrand = E1 * np.exp(-1j * k * (R2[i, j] * np.cos(theta - theta_val)))
            E2[i, j] = simps(simps(integrand, r), theta)

    # Compute intensity distribution
    Ie = np.abs(E2)**2

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
