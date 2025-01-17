import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths

# Background: The Swift-Hohenberg equation is a partial differential equation used to model pattern formation in systems
# undergoing a symmetry-breaking instability. The equation is given by:
# ∂u/∂t = εu - (1 + q0^(-2)∇^2)^2 u - u^3
# where u is the order parameter, ε is a control parameter, and q0 is the critical wavenumber. The pseudo-spectral method
# is a numerical technique that uses Fourier transforms to solve differential equations. It is particularly useful for
# problems with periodic boundary conditions, as it allows for efficient computation in the frequency domain. In this
# method, spatial derivatives are computed in Fourier space, and nonlinear terms are computed in real space. The periodic
# boundary condition implies that the system is treated as if it wraps around, which is naturally handled by the Fourier
# transform. The simulation involves evolving the system from an initial state u0 over a time T with time step dt.



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
    
    # Create the wave numbers for the Fourier space
    kx = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    ky = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    
    # Calculate the square of the wave numbers
    k2 = kx**2 + ky**2
    
    # Precompute the linear operator in Fourier space
    L = epsilon - (1 - q0**(-2) * k2)**2
    
    # Time evolution loop
    for _ in range(num_steps):
        # Transform u to Fourier space
        u_hat = fft2(u)
        
        # Compute the linear part in Fourier space
        u_hat = u_hat * np.exp(L * dt)
        
        # Transform back to real space
        u = np.real(ifft2(u_hat))
        
        # Compute the nonlinear term in real space
        u_nl = u**3
        
        # Transform the nonlinear term to Fourier space
        u_nl_hat = fft2(u_nl)
        
        # Update the Fourier space representation with the nonlinear term
        u_hat = u_hat - dt * u_nl_hat
        
        # Transform back to real space
        u = np.real(ifft2(u_hat))
    
    return u


# Background: The structure factor, Sk, is a measure of how density fluctuations in a system are distributed in reciprocal (Fourier) space. 
# It is particularly useful in the study of pattern formation and phase transitions, as it provides insight into the spatial correlations 
# of the order parameter. In the context of the Swift-Hohenberg equation, the structure factor can reveal the dominant modes and 
# characteristic length scales of the patterns formed. To compute the structure factor, we perform a Fourier transform of the 
# order parameter field, u(x, y), and then calculate the magnitude squared of the resulting Fourier coefficients. The Fourier 
# coordinates, Kx and Ky, are also computed to provide the corresponding wavevector information, which is centered around zero 
# using the fftshift function to facilitate analysis.



def structure_factor(u):
    '''Calculate the structure factor of a 2D real spatial distribution and the Fourier coordinates, shifted to center around k = 0
    Input
    u: order parameter in real space, 2D N*N array of floats
    Output
    Kx: coordinates in k space conjugate to x in real space, 2D N*N array of floats
    Ky: coordinates in k space conjugate to y in real space, 2D N*N array of floats
    Sk: 2D structure factor, 2D array of floats
    '''
    N = u.shape[0]  # Assuming u is a square 2D array of size N*N

    # Perform the Fourier transform of the order parameter field
    u_hat = fft2(u)

    # Calculate the structure factor as the magnitude squared of the Fourier coefficients
    Sk = np.abs(u_hat)**2

    # Shift the zero frequency component to the center
    Sk = fftshift(Sk)

    # Generate the Fourier space coordinates
    kx = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    ky = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    Kx, Ky = np.meshgrid(kx, ky)

    # Shift the coordinates to center around zero
    Kx = fftshift(Kx)
    Ky = fftshift(Ky)

    return Kx, Ky, Sk



# Background: In the context of pattern formation, the structure factor Sk provides insight into the spatial correlations
# of the order parameter. Peaks in Sk indicate dominant modes in the system. To detect the formation of a stripe phase,
# we analyze Sk for peaks that are close to the critical mode q0. A peak is characterized by its height and location in
# k-space. The proximity of a peak to q0 is determined by comparing the radial distance of the peak in k-space to q0,
# within a specified tolerance. The presence of a significant peak near q0 suggests pattern formation, such as stripes,
# in the system. The function will identify such peaks using the find_peaks method, considering both the height and
# proximity to q0.




def analyze_structure_factor(Sk, Kx, Ky, q0, min_height):
    '''Analyze the structure factor to identify peak near q0
    Input:
    Sk: 2D structure factor, 2D array of floats
    Kx: coordinates in k space conjugate to x in real space, 2D N*N array of floats
    Ky: coordinates in k space conjugate to y in real space, 2D N*N array of floats
    q0: critical mode, float
    min_height: threshold height of the peak in the structure factor to be considered, float
    Output:
    peak_found_near_q0: if a peak is found in the structure factor near q0 (meets combined proximity, narrowness, and height criteria), boolean
    peak_near_q0_location: location of the peak near q0 in terms of radial k, float; set to 0 if no peak is found near q0
    '''
    
    # Calculate the radial k values
    radial_k = np.sqrt(Kx**2 + Ky**2)
    
    # Flatten the arrays for peak analysis
    Sk_flat = Sk.flatten()
    radial_k_flat = radial_k.flatten()
    
    # Find peaks in the structure factor
    peaks, properties = find_peaks(Sk_flat, height=min_height)
    
    # Initialize variables to track the peak near q0
    peak_found_near_q0 = False
    peak_location_near_q0 = 0.0
    
    # Tolerance for proximity to q0
    tolerance = 0.1 * q0  # 10% of q0 as a reasonable tolerance
    
    # Analyze each peak
    for peak in peaks:
        peak_k = radial_k_flat[peak]
        if np.abs(peak_k - q0) < tolerance:
            peak_found_near_q0 = True
            peak_location_near_q0 = peak_k
            break  # Stop after finding the first suitable peak
    
    return peak_found_near_q0, peak_location_near_q0


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('55.3', 3)
target = targets[0]

N = 50
q0 = 1.0
u = np.tile(np.sin(q0 * np.arange(N)), (N, 1))
min_height = 1e-12
Kx, Ky, Sk = structure_factor(u)
peak_found_near_q0, peak_near_q0_location = analyze_structure_factor(Sk, Kx, Ky, q0, min_height)
a, b = target
assert a == peak_found_near_q0 and np.allclose(b, peak_near_q0_location)
target = targets[1]

N = 30
q0 = 2.0
i = np.arange(N)[:, np.newaxis]  # Column vector of i indices
j = np.arange(N)  # Row vector of j indices
u = np.sin(q0*i) + np.cos(q0*j)
min_height = 1e-12
Kx, Ky, Sk = structure_factor(u)
peak_found_near_q0, peak_near_q0_location = analyze_structure_factor(Sk, Kx, Ky, q0, min_height)
a, b = target
assert a == peak_found_near_q0 and np.allclose(b, peak_near_q0_location)
target = targets[2]

N = 20
u = np.ones((N, N))
q0 = 1.0
min_height = 1e-12
Kx, Ky, Sk = structure_factor(u)
peak_found_near_q0, peak_near_q0_location = analyze_structure_factor(Sk, Kx, Ky, q0, min_height)
a, b = target
assert a == peak_found_near_q0 and np.allclose(b, peak_near_q0_location)
