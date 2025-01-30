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

    # Calculate the radial wave number magnitude |k| for each point in Kx, Ky
    k_magnitude = np.sqrt(Kx**2 + Ky**2)

    # Find peaks in the structure factor
    Sk_flat = Sk.flatten()
    k_magnitude_flat = k_magnitude.flatten()
    
    # Use scipy's find_peaks to detect peaks in the flattened Sk array
    peaks, properties = find_peaks(Sk_flat, height=min_height)

    # Initialize variables to track the closest peak to q0
    closest_peak_distance = np.inf
    peak_found_near_q0 = False
    peak_location_near_q0 = 0

    # Iterate over detected peaks to find the one closest to q0
    for peak in peaks:
        # Get the wave number magnitude at this peak
        k_peak = k_magnitude_flat[peak]

        # Calculate the distance of this peak from q0
        distance_to_q0 = np.abs(k_peak - q0)

        # If this peak is closer to q0 than any previous, consider it
        if distance_to_q0 < closest_peak_distance:
            closest_peak_distance = distance_to_q0
            peak_location_near_q0 = k_peak

            # Check if the peak is sufficiently close to q0
            tolerance = 0.1 * q0  # Define a tolerance as 10% of q0
            if distance_to_q0 < tolerance:
                peak_found_near_q0 = True

    return peak_found_near_q0, peak_location_near_q0


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e