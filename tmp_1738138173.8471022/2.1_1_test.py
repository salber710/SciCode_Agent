from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps




def simulate_light_diffraction(n, d, RL, R0, lambda_):
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
    # Using the thin lens approximation: 1/f = (n-1) * (1/R - 1/R) with R = RL
    f = RL / (2 * (n - 1))
    phase_change = np.exp(-1j * (2 * np.pi / lambda_) * (R**2 / (2 * f)))

    # Apply the phase change to the initial intensity distribution
    U = I0 * phase_change

    # Compute the intensity distribution on the observation plane
    # Using Fourier optics approach
    Uf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U)))
    Ie = np.abs(Uf)**2

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