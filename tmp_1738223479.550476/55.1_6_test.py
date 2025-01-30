from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths





def solve_SH(u, dt, T, N, epsilon, q0):
    # Define wave numbers using numpy's fft frequencies with a shift
    kx = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    laplacian = -(kx**2 + ky**2)

    # Linear operator L in Fourier space
    L = epsilon - (1 + q0**-2 * laplacian)**2

    # Number of time steps
    num_steps = int(T / dt)

    # Use a modified Euler method
    for _ in range(num_steps):
        # Transform u to Fourier space
        u_hat = fft2(u)

        # Compute the nonlinear term in real space
        nonlinear_real_space = u**3

        # Transform the nonlinear term to Fourier space
        nonlinear_hat = fft2(nonlinear_real_space)

        # Compute the intermediate step using explicit Euler
        u_hat_intermediate = u_hat + dt * (L * u_hat - nonlinear_hat)

        # Transform back to real space for the intermediate step
        u_intermediate = np.real(ifft2(u_hat_intermediate))

        # Compute the nonlinear term at the intermediate step
        nonlinear_intermediate = u_intermediate**3

        # Transform back to Fourier space
        nonlinear_intermediate_hat = fft2(nonlinear_intermediate)

        # Update using a trapezoidal rule for the nonlinear term
        u_hat_new = u_hat + dt * (L * u_hat_new - 0.5 * (nonlinear_hat + nonlinear_intermediate_hat))

        # Inverse Fourier transform to get back to real space
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