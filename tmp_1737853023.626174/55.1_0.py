import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths



# Background: The Swift-Hohenberg equation is a partial differential equation used to model pattern formation in systems undergoing a symmetry-breaking instability. 
# It is often used in the study of convection, chemical reactions, and other systems where spatial patterns emerge. The equation is given by:
# ∂u/∂t = εu - (1 + q0^(-2)∇^2)^2 u - u^3
# where u is the order parameter, ε is a control parameter, and q0 is the critical wavenumber. The pseudo-spectral method is a numerical technique that uses 
# Fourier transforms to solve differential equations. It is particularly useful for problems with periodic boundary conditions, as it allows for efficient computation 
# in the frequency domain. In this method, spatial derivatives are computed in Fourier space, and nonlinear terms are computed in real space. The periodic boundary 
# conditions imply that the system is treated as if it wraps around, which is naturally handled by the Fourier transform.



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
    
    # Create wave numbers for the Fourier space
    kx = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    ky = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    
    # Calculate the square of the wave numbers
    k2 = kx**2 + ky**2
    k4 = k2**2
    
    # Precompute the linear operator in Fourier space
    L = epsilon - (1 - q0**(-2) * k2)**2
    
    # Time-stepping loop
    for _ in range(num_steps):
        # Compute the Fourier transform of the current state
        u_hat = fft2(u)
        
        # Compute the nonlinear term in real space
        nonlinear_term = u**3
        
        # Transform the nonlinear term to Fourier space
        nonlinear_term_hat = fft2(nonlinear_term)
        
        # Update the Fourier transform of u using the pseudo-spectral method
        u_hat = (u_hat + dt * (L * u_hat - nonlinear_term_hat)) / (1.0 + dt * k4)
        
        # Transform back to real space
        u = np.real(ifft2(u_hat))
    
    return u

from scicode.parse.parse import process_hdf5_to_tuple
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
