from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths





def solve_SH(u, dt, T, N, epsilon, q0):
    # Calculate the wave numbers for the Laplacian using numpy's fftfreq
    kx = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    laplacian = -(kx**2 + ky**2)

    # Define the linear operator L in Fourier space
    L = epsilon - (1 + q0**-2 * laplacian)**2

    # Calculate the exponential of the linear operator for use in the ETDRK4 method
    expL_dt = np.exp(dt * L)
    expL_half_dt = np.exp(dt/2 * L)

    # Compute the number of time steps
    num_steps = int(T / dt)

    # Precompute coefficients for the ETDRK4 scheme
    M = 16  # Number of points for complex contour integral
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)  # roots of unity
    LR = dt * L[..., None] + r

    # Calculate the ETDRK4 coefficients
    Q = dt * np.mean((np.exp(LR/2) - 1) / LR, axis=-1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=-1)
    f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1)
    f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1)

    for _ in range(num_steps):
        # Transform u to Fourier space
        u_hat = fft2(u)

        # Nonlinear term in real space
        N_u = u**3

        # Transform the nonlinear term to Fourier space
        N_hat = fft2(N_u)

        # Step 1: Compute the first intermediate step
        a = expL_half_dt * u_hat + Q * N_hat
        Na = np.real(ifft2(a))**3
        Na_hat = fft2(Na)

        # Step 2: Compute the second intermediate step
        b = expL_half_dt * u_hat + Q * Na_hat
        Nb = np.real(ifft2(b))**3
        Nb_hat = fft2(Nb)

        # Step 3: Compute the third intermediate step
        c = expL_dt * u_hat + Q * (2*Nb_hat - N_hat)
        Nc = np.real(ifft2(c))**3
        Nc_hat = fft2(Nc)

        # Final update using the ETDRK4 coefficients
        u_hat = expL_dt * u_hat + N_hat * f1 + 2*(Na_hat + Nb_hat) * f2 + Nc_hat * f3

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