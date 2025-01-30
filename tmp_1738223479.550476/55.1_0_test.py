from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths



# Background: The Swift-Hohenberg equation is a partial differential equation used to study pattern formation 
# in systems undergoing a phase transition. It is typically solved using numerical methods like the pseudo-spectral 
# method, which leverages the efficiency of Fourier transforms. In the pseudo-spectral method, spatial derivatives 
# are calculated in Fourier space, which allows for efficient computation using Fast Fourier Transform (FFT) algorithms. 
# The equation given is:
# 
# ∂u/∂t = εu - (1 + q0^{-2}∇^2)^2 u - u^3
# 
# Here, ε is a control parameter, q0 is the critical wavenumber, ∇^2 is the Laplacian operator, and u is the order 
# parameter. The periodic boundary conditions imply that we can use Fourier transforms to handle spatial derivatives.
# The Laplacian in Fourier space is represented as multiplication by -(k_x^2 + k_y^2), where k_x and k_y are the 
# wave numbers in the x and y directions, respectively. The solution involves transforming the initial state to Fourier 
# space, applying the dynamics in Fourier space, and transforming back to real space.



def solve_SH(u, dt, T, N, epsilon, q0):
    '''Run a 2D simulation of Swift-Hohenberg equation
    Input
    u: initial condition of the order parameter, 2D array of floats
    dt: time step size, float
    T: total time of the evolution, float
    N: system size where the 2D system is of dimension N*N, int
    epsilon: control parameter, float
    q0: critical mode, float
    Output
    u: final state of the system at the end time T.
    '''
    # Calculate the wave numbers for the Laplacian
    kx = fftfreq(N, d=1.0/N) * 2 * np.pi
    ky = fftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    laplacian = -(kx**2 + ky**2)

    # Precompute the linear operator in Fourier space
    L = epsilon - (1 - q0**-2 * laplacian)**2

    # Time evolution loop
    num_steps = int(T / dt)
    for _ in range(num_steps):
        # Transform u to Fourier space
        u_hat = fft2(u)

        # Compute nonlinear term in real space
        non_linear_term = u**3

        # Transform non-linear term to Fourier space
        non_linear_term_hat = fft2(non_linear_term)

        # Update in Fourier space
        u_hat_new = (u_hat + dt * (L * u_hat - non_linear_term_hat))

        # Inverse transform to get back to real space
        u = np.real(ifft2(u_hat_new))

    return u


try:
    targets = process_hdf5_to_tuple('55.1', 3)
    target = targets[0]
    N = 10
    u0 = np.zeros((N, N))
    dt = 0.01
    epsilon = 0
    q0 = 1.
    T = 0.05
    assert np.allclose(solve_SH(u0, dt, T, N, epsilon, q0), target)

    target = targets[1]
    N = 10
    u0 = np.ones((N, N))
    dt = 0.01
    epsilon = 0.7
    q0 = 1.
    T = 0.05
    assert np.allclose(solve_SH(u0, dt, T, N, epsilon, q0), target)

    target = targets[2]
    N = 20
    np.random.seed(1)  # For reproducibility
    u0 = np.random.rand(N, N)
    dt = 0.005
    epsilon = 0.7
    q0 =1.
    T = 0.05
    assert np.allclose(solve_SH(u0, dt, T, N, epsilon, q0), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e