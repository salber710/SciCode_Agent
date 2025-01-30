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

    N = u.shape[0]  # Assuming square grid

    # Compute the Fourier transform of the order parameter field
    u_fft = fft2(u)

    # Calculate the power spectrum (structure factor) as the magnitude squared of the Fourier transform
    Sk = np.abs(u_fft)**2

    # Shift the zero frequency component to the center
    Sk = fftshift(Sk)

    # Compute the wave number grid
    Kx = fftfreq(N, d=1.0/N) * 2.0 * np.pi
    Ky = fftfreq(N, d=1.0/N) * 2.0 * np.pi

    # Shift the wave numbers to center around zero
    Kx = fftshift(Kx)
    Ky = fftshift(Ky)

    # Create 2D grids for Kx and Ky
    Kx, Ky = np.meshgrid(Kx, Ky)

    return Kx, Ky, Sk


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
    K_radial = np.sqrt(Kx**2 + Ky**2)
    
    # Flatten the arrays for easier processing
    Sk_flat = Sk.flatten()
    K_radial_flat = K_radial.flatten()
    
    # Find peaks in the structure factor
    peaks, properties = find_peaks(Sk_flat, height=min_height)
    
    # Initialize variables to track the peak near q0
    peak_found_near_q0 = False
    peak_location_near_q0 = 0
    
    # Define a reasonable proximity threshold for q0, e.g., 10% of q0
    proximity_threshold = 0.1 * q0
    
    # Iterate over the detected peaks
    for peak in peaks:
        # Get the k value for this peak
        k_value = K_radial_flat[peak]
        
        # Check if this peak is within the proximity threshold of q0
        if abs(k_value - q0) <= proximity_threshold:
            peak_found_near_q0 = True
            peak_location_near_q0 = k_value
            break  # Exit loop once a suitable peak is found
    
    return peak_found_near_q0, peak_location_near_q0



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

    # Step 1: Simulate the Swift-Hohenberg equation using the pseudo-spectral method
    u = solve_SH(u0, dt, T, N, epsilon, q0)

    # Step 2: Calculate the structure factor of the final state
    Kx, Ky, Sk = structure_factor(u)

    # Step 3: Analyze the structure factor to detect stripe pattern formation
    if_form_stripes, stripe_mode = analyze_structure_factor(Sk, Kx, Ky, q0, min_height)

    return u, Sk, if_form_stripes, stripe_mode


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e