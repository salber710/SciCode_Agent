from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths



def solve_SH(u, dt, T, N, epsilon, q0):
    # Define wave numbers using numpy's fft frequencies
    kx = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    laplacian = -(kx**2 + ky**2)

    # Define the linear operator in Fourier space
    L = epsilon - (1 + q0**-2 * laplacian)**2

    # Number of time steps
    num_steps = int(T / dt)

    # Initialize Fourier space representation of u
    u_hat = fft2(u)

    # Time-stepping loop using a semi-implicit backward Euler method
    for _ in range(num_steps):
        # Compute the nonlinear term in real space
        nonlinear_term = u**3

        # Transform the nonlinear term to Fourier space
        nonlinear_term_hat = fft2(nonlinear_term)

        # Update in Fourier space using a backward Euler scheme
        u_hat = (u_hat + dt * nonlinear_term_hat) / (1 - dt * L)

        # Inverse transform to get back to real space
        u = np.real(ifft2(u_hat))

    return u





def structure_factor(u):
    '''Calculate the structure factor of a 2D real spatial distribution and the Fourier coordinates, shifted to center around k = 0.
    Input
    u: order parameter in real space, 2D N*N array of floats
    Output
    Kx: coordinates in k space conjugate to x in real space, 2D N*N array of floats
    Ky: coordinates in k space conjugate to y in real space, 2D N*N array of floats
    Sk: 2D structure factor, 2D array of floats
    '''
    
    N = u.shape[0]  # Assuming u is an N by N array
    
    # Perform a 2D Fourier transform using fft2
    u_hat = fft2(u)

    # Compute the power spectrum using the squared real and imaginary parts
    Sk = (u_hat.real**2 + u_hat.imag**2)

    # Shift the zero-frequency component to the center of the spectrum
    Sk_shifted = fftshift(Sk)

    # Calculate wave number coordinates in k-space
    kx = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='xy')

    return Kx, Ky, Sk_shifted


try:
    targets = process_hdf5_to_tuple('55.2', 3)
    target = targets[0]
    N = 20
    u = np.tile(np.sin(np.arange(N)), (N, 1))
    assert np.allclose(structure_factor(u), target)

    target = targets[1]
    N = 30
    i = np.arange(N)[:, np.newaxis]  # Column vector of i indices
    j = np.arange(N)  # Row vector of j indices
    u = np.sin(i) + np.cos(j)
    assert np.allclose(structure_factor(u), target)

    target = targets[2]
    N = 20
    u = np.ones((N, N))
    assert np.allclose(structure_factor(u), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e