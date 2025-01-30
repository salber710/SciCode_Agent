from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths



# Background: The Swift-Hohenberg equation is a partial differential equation used to model pattern formation in various physical systems. 
# It is often solved using spectral methods due to their efficiency in handling differential operators. 
# The pseudo-spectral method involves transforming the spatial domain into the frequency domain using the Fourier transform, 
# which simplifies the application of differential operators. The equation given is:
# ∂u/∂t = εu - (1 + q0^(-2)∇^2)^2 u - u^3
# Here, ε is a control parameter, q0 is a critical mode, and ∇^2 is the Laplacian operator. 
# The periodic boundary condition implies that the system is treated as if it wraps around, 
# which is naturally handled by the Fourier transform. The time evolution is computed by discretizing time with a step size dt 
# and iterating until the total time T is reached. The order parameter u is kept real by using real-to-complex and complex-to-real FFTs.



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
    
    # Create the wave numbers for the Fourier transform
    kx = rfftfreq(N) * 2 * np.pi
    ky = np.fft.fftfreq(N) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    
    # Calculate the Laplacian in Fourier space
    laplacian = -(kx**2 + ky**2)
    
    # Calculate the linear operator in Fourier space
    L = epsilon - (1 + (laplacian / q0**2))**2
    
    # Time-stepping loop
    for _ in range(num_steps):
        # Transform u to Fourier space
        u_hat = rfft2(u)
        
        # Nonlinear term in real space
        nonlinear_term = u**3
        
        # Transform the nonlinear term to Fourier space
        nonlinear_term_hat = rfft2(nonlinear_term)
        
        # Update u_hat using the pseudo-spectral method
        u_hat = (u_hat + dt * (L * u_hat - nonlinear_term_hat)) / (1 - dt * L)
        
        # Transform back to real space
        u = irfft2(u_hat)
    
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