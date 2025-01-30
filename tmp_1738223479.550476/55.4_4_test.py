from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, rfft2, irfft2, fftfreq, rfftfreq
from scipy.signal import find_peaks, peak_widths



def solve_SH(u, dt, T, N, epsilon, q0):
    # Define wave numbers using numpy's fft frequencies
    kx = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    laplacian = -(kx**2 + ky**2)

    # Define the linear operator in Fourier space
    L = epsilon - (1 + q0**-2 * laplacian)**2

    # Number of time steps
    num_steps = int(T / dt)

    # Initialize Fourier space representation of u
    u_hat = fft2(u)

    # Time-stepping loop using a semi-implicit backward Euler method
    for _ in range(num_steps):
        # Compute the nonlinear term in real space
        nonlinear_term = u**3

        # Transform the nonlinear term to Fourier space
        nonlinear_term_hat = fft2(nonlinear_term)

        # Update in Fourier space using a backward Euler scheme
        u_hat = (u_hat + dt * nonlinear_term_hat) / (1 - dt * L)

        # Inverse transform to get back to real space
        u = np.real(ifft2(u_hat))

    return u




def structure_factor(u):
    '''Calculate the structure factor of a 2D real spatial distribution and the Fourier coordinates, shifted to center around k = 0.
    Input
    u: order parameter in real space, 2D N*N array of floats
    Output
    Kx: coordinates in k space conjugate to x in real space, 2D N*N array of floats
    Ky: coordinates in k space conjugate to y in real space, 2D N*N array of floats
    Sk: 2D structure factor, 2D array of floats
    '''
    
    N = u.shape[0]  # Assuming u is an N by N array
    
    # Perform a 2D Fourier transform using fft2
    u_hat = fft2(u)

    # Compute the power spectrum using the squared real and imaginary parts
    Sk = (u_hat.real**2 + u_hat.imag**2)

    # Shift the zero-frequency component to the center of the spectrum
    Sk_shifted = fftshift(Sk)

    # Calculate wave number coordinates in k-space
    kx = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='xy')

    return Kx, Ky, Sk_shifted




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

    # Apply a uniform filter to smooth the structure factor, emphasizing larger features
    Sk_smooth = uniform_filter(Sk, size=3)

    # Find locations where Sk is a local maximum compared to its direct neighbors
    local_maxima = (Sk_smooth > np.roll(Sk_smooth, 1, axis=0)) & (Sk_smooth > np.roll(Sk_smooth, -1, axis=0)) & \
                   (Sk_smooth > np.roll(Sk_smooth, 1, axis=1)) & (Sk_smooth > np.roll(Sk_smooth, -1, axis=1))

    # Filter out the maxima that do not meet the minimum height requirement
    significant_maxima = local_maxima & (Sk > min_height)

    # Extract radial k values corresponding to significant maxima
    significant_k_values = radial_k[significant_maxima]

    # If no significant peaks are found, return defaults
    if len(significant_k_values) == 0:
        return False, 0.0

    # Calculate distances to q0 from significant peaks
    distances_to_q0 = np.abs(significant_k_values - q0)

    # Find the minimum distance and its corresponding index
    min_distance_index = np.argmin(distances_to_q0)

    # Determine if the closest peak is sufficiently close to q0
    tolerance = 0.02 * q0  # Define a different tolerance for proximity
    peak_found_near_q0 = distances_to_q0[min_distance_index] < tolerance
    peak_near_q0_location = significant_k_values[min_distance_index] if peak_found_near_q0 else 0.0

    return peak_found_near_q0, peak_near_q0_location



def SH_pattern_formation(u0, dt, T, N, epsilon, q0, min_height):



    def solve_SH(u, dt, T, N, epsilon, q0):
        # Precompute the wave numbers in Fourier space
        kx = fftfreq(N, d=1.0/N) * 2 * np.pi
        ky = fftfreq(N, d=1.0/N) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        laplacian = -(kx**2 + ky**2)

        # Define the linear operator in Fourier space
        L = epsilon - (1 + q0**-2 * laplacian)**2

        # Number of time steps
        num_steps = int(T / dt)

        # Transform the initial condition to Fourier space
        u_hat = fft2(u)

        # Use a Crank-Nicolson method for time-stepping
        for _ in range(num_steps):
            # Compute the nonlinear term in real space
            u_real = np.real(ifft2(u_hat))
            nonlinear_term = u_real**3

            # Transform the nonlinear term to Fourier space
            nonlinear_term_hat = fft2(nonlinear_term)

            # Update in Fourier space using Crank-Nicolson
            u_hat = (u_hat + 0.5 * dt * nonlinear_term_hat) / (1 - 0.5 * dt * L)

        # Return the final state in real space
        return np.real(ifft2(u_hat))

    def structure_factor(u):
        # Perform a 2D Fourier transform and compute the power spectrum
        u_hat = fft2(u)
        Sk = np.abs(u_hat)**2
        return fftshift(Sk)

    def analyze_structure_factor(Sk, q0, min_height):
        # Flatten the structure factor for easier analysis
        Sk_flat = Sk.flatten()
        max_Sk_index = np.argmax(Sk_flat)

        if Sk_flat[max_Sk_index] < min_height:
            return False, 0.0

        # Calculate wave numbers corresponding to the indices
        kx = fftfreq(N, d=1.0/N) * 2 * np.pi
        ky = fftfreq(N, d=1.0/N) * 2 * np.pi
        Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
        Kx = fftshift(Kx).flatten()
        Ky = fftshift(Ky).flatten()

        # Get the wavenumber of the peak
        peak_kx, peak_ky = Kx[max_Sk_index], Ky[max_Sk_index]
        peak_k = np.sqrt(peak_kx**2 + peak_ky**2)

        # Determine if the peak is close to the critical mode
        if np.abs(peak_k - q0) < 0.02 * q0:
            return True, peak_k
        else:
            return False, 0.0

    # Simulate the Swift-Hohenberg equation
    u_final = solve_SH(u0, dt, T, N, epsilon, q0)

    # Compute the structure factor of the final state
    Sk = structure_factor(u_final)

    # Analyze the structure factor for stripe formation
    if_form_stripes, stripe_mode = analyze_structure_factor(Sk, q0, min_height)

    return u_final, Sk, if_form_stripes, stripe_mode


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