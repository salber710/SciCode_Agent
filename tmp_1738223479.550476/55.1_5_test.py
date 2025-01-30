from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths





def solve_SH(u, dt, T, N, epsilon, q0):
    # Define wave numbers using fft frequencies
    k = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky = np.meshgrid(k, k)
    laplacian = -(kx**2 + ky**2)

    # Linear operator in Fourier space
    L = epsilon - (1 + q0**-2 * laplacian)**2

    # Number of steps
    num_steps = int(T / dt)

    # Using a second-order Adams-Bashforth method for time-stepping
    u_hat_prev = fft2(u)
    nonlinear_prev = fft2(u**3)

    for _ in range(num_steps):
        # Transform current state to Fourier space
        u_hat = fft2(u)

        # Compute the nonlinear term in real space
        nonlinear_real_space = u**3

        # Transform the nonlinear term to Fourier space
        nonlinear_hat = fft2(nonlinear_real_space)

        # Adams-Bashforth update in Fourier space
        if _ == 0:
            # Use Euler step for first iteration
            u_hat_new = u_hat + dt * (L * u_hat - nonlinear_hat)
        else:
            # Use previous step's nonlinear term for Adams-Bashforth
            u_hat_new = u_hat_prev + (3/2) * dt * (L * u_hat - nonlinear_hat) - (1/2) * dt * (L * u_hat_prev - nonlinear_prev)

        # Update previous variables
        u_hat_prev = u_hat
        nonlinear_prev = nonlinear_hat

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