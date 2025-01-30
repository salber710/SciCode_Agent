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
    
    if dt <= 0:
        raise ValueError("Time step size dt must be positive.")
    
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    
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



# Background: The structure factor, Sk, is a measure of how density fluctuations in a system are distributed in reciprocal space (k-space). 
# It is often used in the study of spatial patterns and phase transitions. In the context of the Swift-Hohenberg equation, the structure factor 
# can provide insights into the dominant spatial frequencies of the order parameter field u(x, y). The Fourier transform of the order parameter 
# is used to compute Sk, which is essentially the power spectrum of the field. The coordinates in k-space, Kx and Ky, are derived from the 
# Fourier transform and are centered around zero using a shift operation to facilitate analysis of the spatial frequency content.



def structure_factor(u):
    '''Calculate the structure factor of a 2D real spatial distribution and the Fourier coordinates, shifted to center around k = 0
    Input
    u: order parameter in real space, 2D N*N array of floats
    Output
    Kx: coordinates in k space conjugate to x in real space, 2D N*N array of floats
    Ky: coordinates in k space conjugate to y in real space, 2D N*N array of floats
    Sk: 2D structure factor, 2D array of floats
    '''
    
    # Get the size of the input array
    N = u.shape[0]
    
    # Compute the Fourier transform of the order parameter field
    u_hat = fft2(u)
    
    # Compute the structure factor as the squared magnitude of the Fourier transform
    Sk = np.abs(u_hat)**2
    
    # Shift the zero frequency component to the center of the spectrum
    Sk = fftshift(Sk)
    
    # Generate the k-space coordinates
    kx = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    ky = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    Kx, Ky = np.meshgrid(kx, ky)
    
    # Shift the k-space coordinates to center around zero
    Kx = fftshift(Kx)
    Ky = fftshift(Ky)
    
    return Kx, Ky, Sk

from scicode.parse.parse import process_hdf5_to_tuple
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
