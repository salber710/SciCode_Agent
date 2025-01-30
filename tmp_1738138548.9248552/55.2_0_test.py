from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths



def solve_SH(u, dt, T, N, epsilon, q0):
    num_steps = int(T / dt)
    k = 2 * np.pi * np.fft.fftfreq(N)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    laplacian = -(kx**2 + ky**2)
    linear_term = epsilon - (1 + laplacian / q0**2)**2

    for _ in range(num_steps):
        u_hat = fftn(u)
        nonlinear_term = u**3
        nonlinear_term_hat = fftn(nonlinear_term)
        u_hat = u_hat + dt * (linear_term * u_hat - nonlinear_term_hat)
        u = np.real(ifftn(u_hat))

    return u



# Background: The structure factor, Sk, is a measure of how density fluctuations in a system are distributed in reciprocal space (k-space). 
# It is calculated as the magnitude squared of the Fourier transform of the order parameter field, u(x, y). 
# The Fourier transform converts the spatial distribution into frequency space, where each point represents a wave vector (k_x, k_y) 
# corresponding to a particular spatial frequency. The structure factor is useful in identifying patterns and periodicities in the system, 
# such as those arising from instabilities or phase transitions. In this context, we use the 2D Fourier transform to compute Sk, 
# and we shift the zero frequency component to the center of the spectrum for better visualization and analysis.



def structure_factor(u):
    '''Calculate the structure factor of a 2D real spatial distribution and the Fourier coordinates, shifted to center around k = 0
    Input
    u: order parameter in real space, 2D N*N array of floats
    Output
    Kx: coordinates in k space conjugate to x in real space, 2D N*N array of floats
    Ky: coordinates in k space conjugate to y in real space, 2D N*N array of floats
    Sk: 2D structure factor, 2D array of floats
    '''
    N = u.shape[0]
    
    # Compute the 2D Fourier transform of the order parameter field
    u_hat = fft2(u)
    
    # Compute the structure factor as the magnitude squared of the Fourier transform
    Sk = np.abs(u_hat)**2
    
    # Shift the zero frequency component to the center
    Sk = fftshift(Sk)
    
    # Generate the k-space coordinates
    k = fftfreq(N) * 2 * np.pi
    Kx, Ky = np.meshgrid(k, k, indexing='ij')
    
    # Shift the k-space coordinates to center around zero
    Kx = fftshift(Kx)
    Ky = fftshift(Ky)
    
    return Kx, Ky, Sk


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