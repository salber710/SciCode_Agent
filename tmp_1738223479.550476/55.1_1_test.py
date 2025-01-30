from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths





def solve_SH(u, dt, T, N, epsilon, q0):
    # Define the wave numbers with fftshift for proper filtering
    kx = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    laplacian = -(kx**2 + ky**2)

    # Compute the linear operator in Fourier space
    L = epsilon - (1 + q0**-2 * laplacian)**2

    # Precompute the exponential of the linear operator for time-stepping
    expL = np.exp(dt * L)
    expL2 = np.exp(dt/2 * L)

    # Number of time steps
    num_steps = int(T / dt)
    for _ in range(num_steps):
        # Transform u to Fourier space
        u_hat = fft2(u)

        # Compute nonlinear term in real space
        non_linear_term = u**3

        # Transform non-linear term to Fourier space
        non_linear_term_hat = fft2(non_linear_term)

        # Apply exponential time differencing (ETD) scheme
        u_hat = expL * u_hat + dt * expL2 * non_linear_term_hat

        # Inverse transform to get back to real space
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