from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths

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
    # Calculate the number of time steps
    num_steps = int(T / dt)
    
    # Compute the wave number grid
    kx = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    ky = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    
    # Create a 2D grid of wave numbers
    kx, ky = np.meshgrid(kx, ky)
    
    # Compute the square of the wave numbers
    k_squared = kx**2 + ky**2
    
    # Precompute the linear operator in Fourier space
    L = epsilon - (1 + (k_squared / q0**2))**2
    
    # Time evolution loop
    for step in range(num_steps):
        # Compute the Fourier transform of the current state
        u_hat = fft2(u)
        
        # Apply the linear operator in Fourier space
        u_hat = u_hat * (1 + dt * L)
        
        # Transform back to real space
        u = np.real(ifft2(u_hat))
        
        # Apply the nonlinear term
        u = u - dt * u**3
    
    return u





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
    
    # Compute the Fourier transform of the order parameter
    u_hat = fft2(u)
    
    # Compute the structure factor
    Sk = np.abs(u_hat)**2
    
    # Shift zero frequency component to the center
    Sk = fftshift(Sk)
    
    # Compute wave number grid
    kx = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    ky = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    
    # Create a 2D grid of wave numbers and shift the zero frequency component to the center
    Kx, Ky = np.meshgrid(kx, ky)
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