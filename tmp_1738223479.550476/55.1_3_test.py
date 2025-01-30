from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths





def solve_SH(u, dt, T, N, epsilon, q0):
    # Define the wave numbers using numpy's fftfreq
    k = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky = np.meshgrid(k, k)
    laplacian = -(kx**2 + ky**2)

    # Compute the linear operator in Fourier space
    L = epsilon - (1 + q0**-2 * laplacian)**2

    # Time evolution loop
    num_steps = int(T / dt)
    for step in range(num_steps):
        # Transform u to Fourier space
        u_hat = fft2(u)

        # Compute the cubic nonlinear term in real space
        u_cubed = u**3

        # Transform the nonlinear term to Fourier space
        u_cubed_hat = fft2(u_cubed)

        # Update u_hat using a mix of explicit and implicit scheme
        u_hat = (u_hat + dt * (L * u_hat)) / (1 + dt * u_cubed_hat)

        # Inverse Fourier transform to obtain the real space representation
        u = np.real(ifft2(u_hat))

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