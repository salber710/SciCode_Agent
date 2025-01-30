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