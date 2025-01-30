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
    u: order parameter in real space, 2D array of floats
    Output
    Kx: coordinates in k space conjugate to x in real space, 2D array of floats
    Ky: coordinates in k space conjugate to y in real space, 2D array of floats
    Sk: 2D structure factor, 2D array of floats
    '''
    
    # Get the size of the input array
    N, M = u.shape
    
    # Compute the Fourier transform of the order parameter field
    u_hat = fft2(u)
    
    # Compute the structure factor as the squared magnitude of the Fourier transform
    Sk = np.abs(u_hat)**2
    
    # Shift the zero frequency component to the center of the spectrum
    Sk = fftshift(Sk)
    
    # Generate the k-space coordinates
    kx = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    ky = fftfreq(M, d=1.0/M) * 2.0 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    # Shift the k-space coordinates to center around zero
    Kx = fftshift(Kx)
    Ky = fftshift(Ky)
    
    # Normalize Sk by the total number of elements to match the Parseval's theorem
    Sk /= (N * M)
    
    return Kx, Ky, Sk


# Background: In the context of pattern formation, the structure factor (Sk) provides insight into the spatial frequency content of a system. 
# Peaks in the structure factor indicate dominant modes of spatial variation. For the Swift-Hohenberg equation, a peak near the critical mode 
# q0 suggests the emergence of a pattern, such as stripes. To detect such a pattern, we analyze the structure factor to find peaks and check 
# if any are close to q0. The proximity to q0 is determined by a tolerance, and the peak must also exceed a minimum height to be considered 
# significant. This analysis helps identify whether the system has undergone a symmetry-breaking instability leading to pattern formation.



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
    
    # Flatten the arrays for easier peak analysis
    Sk_flat = Sk.flatten()
    radial_k_flat = radial_k.flatten()
    
    # Find peaks in the structure factor
    peaks, properties = find_peaks(Sk_flat, height=min_height)
    
    # Initialize variables to track the peak near q0
    peak_found_near_q0 = False
    peak_near_q0_location = 0.0
    
    # Define a reasonable tolerance for proximity to q0
    tolerance = 0.1 * q0
    
    # Analyze each peak to see if it is near q0
    for peak in peaks:
        peak_k = radial_k_flat[peak]
        if np.abs(peak_k - q0) < tolerance:
            peak_found_near_q0 = True
            peak_near_q0_location = peak_k
            break  # Stop after finding the first suitable peak
    
    return peak_found_near_q0, peak_near_q0_location



# Background: The Swift-Hohenberg equation is a model for pattern formation in systems undergoing symmetry-breaking instabilities.
# It is often used to study phenomena such as convection and chemical reactions. The pseudo-spectral method is a numerical technique
# that leverages Fourier transforms to efficiently solve differential equations with periodic boundary conditions. In this context,
# the structure factor (Sk) is a measure of the spatial frequency content of the system, and peaks in Sk indicate dominant modes
# of spatial variation. A peak near the critical mode q0 suggests the emergence of a pattern, such as stripes. By simulating the
# Swift-Hohenberg equation and analyzing the structure factor, we can determine if a stripe pattern forms and identify its wavenumber.




def SH_pattern_formation(u0, dt, T, N, epsilon, q0, min_height):
    '''This function simulates the time evolution of the Swift-Hohenberg equation using the pseudo-spectral method,
    computes the structure factor of the final state, and analyze the structure factor to identify pattern formation.
    Input
    u: initial condition of the order parameter, 2D array of floats
    dt: time step size, float
    T: total time of the evolution, float
    N: system size where the 2D system is of dimension N*N, int
    epsilon: control parameter, float
    q0: critical mode, float
    min_height: threshold height of the peak in the structure factor to be considered, float; set to 0 if no stripe is formed
    Output
    u: spatial-temporal distribution at time T, 2D array of float
    Sk: structure factor of the final states, 2D array of float
    if_form_stripes: if the system form stripe pattern, boolean
    stripe_mode: the wavenumber of the strips, float
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
    
    # Initialize the order parameter
    u = u0.copy()
    
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
    
    # Compute the structure factor of the final state
    u_hat_final = fft2(u)
    Sk = np.abs(u_hat_final)**2
    Sk = fftshift(Sk)
    
    # Generate the k-space coordinates
    Kx = fftshift(kx)
    Ky = fftshift(ky)
    
    # Normalize Sk by the total number of elements to match Parseval's theorem
    Sk /= (N * N)
    
    # Analyze the structure factor to identify pattern formation
    radial_k = np.sqrt(Kx**2 + Ky**2)
    Sk_flat = Sk.flatten()
    radial_k_flat = radial_k.flatten()
    
    # Find peaks in the structure factor
    peaks, _ = find_peaks(Sk_flat, height=min_height)
    
    # Initialize variables to track the peak near q0
    if_form_stripes = False
    stripe_mode = 0.0
    
    # Define a reasonable tolerance for proximity to q0
    tolerance = 0.1 * q0
    
    # Analyze each peak to see if it is near q0
    for peak in peaks:
        peak_k = radial_k_flat[peak]
        if np.abs(peak_k - q0) < tolerance:
            if_form_stripes = True
            stripe_mode = peak_k
            break  # Stop after finding the first suitable peak
    
    return u, Sk, if_form_stripes, stripe_mode

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('55.4', 6)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
np.random.seed(42)  # For reproducibility
N = 100
u0 = np.random.rand(N, N)
dt = 0.005
T = 50.
epsilon = 0.7
q0 = 0.5
min_height = 1e-12
assert cmp_tuple_or_list(SH_pattern_formation(u0, dt, T, N, epsilon, q0, min_height), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
np.random.seed(42)  # For reproducibility
N = 100
u0 = np.random.rand(N, N)
dt = 0.005
T = 50.
epsilon = 0.2
q0 = 0.5
min_height = 1e-12
assert cmp_tuple_or_list(SH_pattern_formation(u0, dt, T, N, epsilon, q0, min_height), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
np.random.seed(42)  # For reproducibility
N = 100
u0 = np.random.rand(N, N)
dt = 0.005
T = 50.
epsilon = - 0.5
q0 = 0.5
min_height = 1e-12
assert cmp_tuple_or_list(SH_pattern_formation(u0, dt, T, N, epsilon, q0, min_height), target)
target = targets[3]

from scicode.compare.cmp import cmp_tuple_or_list
np.random.seed(42)  # For reproducibility
N = 200
u0 = np.random.rand(N, N)
dt = 0.005
T = 50.
epsilon = 0.7
q0 = 0.4
min_height = 1e-12
assert cmp_tuple_or_list(SH_pattern_formation(u0, dt, T, N, epsilon, q0, min_height), target)
target = targets[4]

from scicode.compare.cmp import cmp_tuple_or_list
np.random.seed(42)  # For reproducibility
N = 200
u0 = np.random.rand(N, N)
dt = 0.005
T = 50.
epsilon = 0.7
q0 = 1.0
min_height = 1e-12
assert cmp_tuple_or_list(SH_pattern_formation(u0, dt, T, N, epsilon, q0, min_height), target)
target = targets[5]

from scicode.compare.cmp import cmp_tuple_or_list
np.random.seed(42)  # For reproducibility
N = 200
u0 = np.random.rand(N, N)
dt = 0.005
T = 50.
epsilon = 0.7
q0 = 2.0
min_height = 1e-12
assert cmp_tuple_or_list(SH_pattern_formation(u0, dt, T, N, epsilon, q0, min_height), target)
