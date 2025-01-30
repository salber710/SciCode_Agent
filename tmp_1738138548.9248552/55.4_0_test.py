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




def structure_factor(u):
    N = u.shape[0]
    
    # Compute the 2D Fourier transform of the order parameter field
    u_hat = fft2(u)
    
    # Compute the structure factor as the magnitude squared of the Fourier transform
    Sk = np.abs(u_hat)**2
    
    # Normalize the structure factor by the sum of all elements to get relative intensities
    Sk /= np.sum(Sk)
    
    # Generate the k-space coordinates
    k = fftfreq(N) * N * 2 * np.pi
    Kx, Ky = np.meshgrid(k, k, indexing='ij')
    
    # Shift the zero frequency component to the center
    Kx = fftshift(Kx)
    Ky = fftshift(Ky)
    Sk = fftshift(Sk)
    
    return Kx, Ky, Sk



def analyze_structure_factor(Sk, Kx, Ky, q0, min_height):
    # Calculate the radial k values
    radial_k = np.sqrt(Kx**2 + Ky**2)
    
    # Find indices where Sk is above the minimum height
    significant_indices = np.where(Sk > min_height)
    
    # Extract the corresponding k values
    significant_k_values = radial_k[significant_indices]
    
    # Define a tolerance for proximity to q0
    tolerance = 0.05 * q0
    
    # Find peaks near q0 within the tolerance and above the minimum height
    close_peaks = significant_k_values[np.abs(significant_k_values - q0) <= tolerance]
    
    # Determine if there is a peak and find the highest peak if multiple
    if close_peaks.size > 0:
        highest_peak = close_peaks[np.argmax(Sk[significant_indices][np.abs(significant_k_values - q0) <= tolerance])]
        return True, highest_peak
    else:
        return False, 0



# Background: The Swift-Hohenberg equation is a partial differential equation used to model pattern formation in systems
# undergoing a symmetry-breaking instability. The pseudo-spectral method is a numerical technique that leverages the
# Fourier transform to efficiently solve differential equations by transforming them into algebraic equations in
# Fourier space. The structure factor is a measure of the intensity of different spatial frequencies in a system,
# and peaks in the structure factor indicate dominant patterns or modes. In this context, we are interested in
# identifying whether a stripe pattern forms, which corresponds to a peak in the structure factor near the critical
# mode q0. The function will simulate the system, compute the structure factor, and analyze it to detect pattern
# formation.



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
    
    # Simulate the Swift-Hohenberg equation
    num_steps = int(T / dt)
    k = 2 * np.pi * np.fft.fftfreq(N)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    laplacian = -(kx**2 + ky**2)
    linear_term = epsilon - (1 + laplacian / q0**2)**2
    u = u0.copy()

    for _ in range(num_steps):
        u_hat = fft2(u)
        nonlinear_term = u**3
        nonlinear_term_hat = fft2(nonlinear_term)
        u_hat = u_hat + dt * (linear_term * u_hat - nonlinear_term_hat)
        u = np.real(ifft2(u_hat))

    # Compute the structure factor
    u_hat = fft2(u)
    Sk = np.abs(u_hat)**2
    Sk /= np.sum(Sk)  # Normalize the structure factor

    # Generate the k-space coordinates
    k = fftfreq(N) * N * 2 * np.pi
    Kx, Ky = np.meshgrid(k, k, indexing='ij')
    Kx = fftshift(Kx)
    Ky = fftshift(Ky)
    Sk = fftshift(Sk)

    # Analyze the structure factor for pattern formation
    radial_k = np.sqrt(Kx**2 + Ky**2)
    significant_indices = np.where(Sk > min_height)
    significant_k_values = radial_k[significant_indices]
    tolerance = 0.05 * q0
    close_peaks = significant_k_values[np.abs(significant_k_values - q0) <= tolerance]

    if close_peaks.size > 0:
        highest_peak = close_peaks[np.argmax(Sk[significant_indices][np.abs(significant_k_values - q0) <= tolerance])]
        return u, Sk, True, highest_peak
    else:
        return u, Sk, False, 0


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