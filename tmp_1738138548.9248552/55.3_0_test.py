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



# Background: In the context of pattern formation, the structure factor (Sk) is a crucial tool for identifying the presence of spatial patterns in a system. It is derived from the Fourier transform of the order parameter field and provides information about the distribution of spatial frequencies (or modes) in the system. A peak in the structure factor indicates a dominant mode, which corresponds to a regular pattern in real space. The critical mode, q0, is a specific wave number where pattern formation is expected to occur. To detect the formation of a stripe phase, we need to identify if there is a significant peak in the structure factor near this critical mode. This involves checking if the peak's magnitude exceeds a certain threshold (min_height) and if its location in k-space is close to q0 within a reasonable tolerance.



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
    
    # Flatten the arrays for easier processing
    Sk_flat = Sk.flatten()
    radial_k_flat = radial_k.flatten()
    
    # Find peaks in the structure factor
    peaks, properties = find_peaks(Sk_flat, height=min_height)
    
    # Initialize variables to track the peak near q0
    peak_found_near_q0 = False
    peak_location_near_q0 = 0
    
    # Define a reasonable tolerance for proximity to q0
    tolerance = 0.1 * q0
    
    # Check each peak to see if it is near q0
    for peak in peaks:
        peak_k = radial_k_flat[peak]
        if np.abs(peak_k - q0) < tolerance:
            peak_found_near_q0 = True
            peak_location_near_q0 = peak_k
            break
    
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