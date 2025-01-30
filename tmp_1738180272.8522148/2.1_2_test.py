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
    x = np.linspace(-R0, R0, mr2)
    y = np.linspace(-R0, R0, mr2)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Calculate the initial Gaussian beam intensity distribution
    I0 = np.exp(-2 * (R / R0)**2)

    # Calculate the phase change due to the lens
    # Using a parabolic approximation for the lens shape
    f = RL / (2 * (n - 1))
    phase_change = np.exp(-1j * (np.pi / lambda_) * (R**2 / (2 * f)))

    # Apply the phase change to the initial intensity distribution
    U = I0 * phase_change

    # Compute the intensity distribution on the observation plane using angular spectrum method
    k = 2 * np.pi / lambda_
    fx = np.fft.fftfreq(mr2, d=(x[1] - x[0]))
    fy = np.fft.fftfreq(mr2, d=(y[1] - y[0]))
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * lambda_ * (FX**2 + FY**2) * d)
    Uf = np.fft.fft2(U)
    U_propagated = np.fft.ifft2(Uf * H)
    Ie = np.abs(U_propagated)**2

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