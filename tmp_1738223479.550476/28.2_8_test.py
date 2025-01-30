from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps


def propagate_gaussian_beam(N, Ld, w0, z, L):
    """
    Propagate a Gaussian beam and calculate its intensity distribution before and after propagation.
    Input
    N : int
        The number of sampling points in each dimension (assumes a square grid).
    Ld : float
        Wavelength of the Gaussian beam.
    w0 : float
        Waist radius of the Gaussian beam at its narrowest point.
    z : float
        Propagation distance of the Gaussian beam.
    L : float
        Side length of the square area over which the beam is sampled.
    Output
    Gau:     a 2D array with dimensions (N+1, N+1) representing the absolute value of the beam's amplitude distribution before propagation.
    Gau_Pro: a 2D array with dimensions (N+1, N+1) representing the absolute value of the beam's amplitude distribution after propagation.
    """
    
    # Define spatial grid using a unique angle-based construction
    x = np.linspace(-L/2, L/2, N+1)
    y = np.linspace(-L/2, L/2, N+1)
    X, Y = np.meshgrid(x, y)

    # Initial Gaussian beam with an arbitrary phase shift for uniqueness
    r_squared = X**2 + Y**2
    Gau = np.exp(-r_squared / (2 * w0**2)) * np.exp(1j * r_squared / (3 * w0**2))

    # Fourier Transform to frequency domain using numpy's rfft2 for a real-valued input optimization
    Gau_ft = np.fft.rfft2(Gau)

    # Define frequency domain coordinates using a unique scaling factor
    dx = L / (N + 1)
    fx = np.fft.rfftfreq(N+1, d=dx)
    fy = np.fft.fftfreq(N+1, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    f_squared = FX**2 + FY**2

    # Calculate propagation phase factor with a novel approach involving hyperbolic functions
    k = 2 * np.pi / Ld
    phase_factor = np.exp(-1j * k * z * np.cosh(f_squared / (4 * np.pi**2)))

    # Apply phase factor in frequency domain
    Gau_ft_propagated = Gau_ft * phase_factor

    # Inverse Fourier Transform to spatial domain using numpy's irfft2 to match rfft2
    Gau_Pro = np.fft.irfft2(Gau_ft_propagated, s=(N+1, N+1))

    # Return the absolute value of the distributions
    return np.abs(Gau), np.abs(Gau_Pro)




def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    '''Calculates the Gaussian beam waist size using an alternative approach that differs from previous implementations.'''
    
    # Calculate the Rayleigh range and initial complex beam parameter q0
    zR = np.pi * w0**2 / wavelength
    q0 = s + 1j * zR
    
    # Extract ABCD matrix components using a different method
    A, B, C, D = Mf1.flatten()

    # Transform the beam parameter through the lens with a unique calculation
    q_lens = (A * q0 + B) / (C * q0 + D)

    # Propagate the beam parameter over distance L1
    q_propagated = q_lens + L1
    
    # Compute the waist sizes using a distinct method
    wz = np.empty_like(z, dtype=float)
    for idx, z_val in enumerate(z):
        q_current = q_propagated + z_val
        inv_q_real = np.real(1 / q_current)
        inv_q_imag = np.imag(1 / q_current)
        
        # Use a unique conditional handling for the calculation
        if inv_q_real == 0 and inv_q_imag == 0:
            wz[idx] = np.inf
        else:
            wz[idx] = np.sqrt(-wavelength / (np.pi * inv_q_imag)) if inv_q_imag != 0 else np.inf

    return wz


try:
    targets = process_hdf5_to_tuple('28.2', 3)
    target = targets[0]
    lambda_ = 1.064e-3  # Wavelength in mm
    w0 = 0.2  # Initial waist size in mm
    R0 = 1.0e30  # Initial curvature radius
    Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
    L1 = 200  # Distance in mm
    z = np.linspace(0, 300, 1000)  # Array of distances in mm
    s = 150 # initial waist position
    # User input for beam quality index
    Mp2 = 1.5
    assert np.allclose(gaussian_beam_through_lens(lambda_, w0, R0, Mf1, z,Mp2,L1,s), target)

    target = targets[1]
    lambda_ = 1.064e-3  # Wavelength in mm
    w0 = 0.2  # Initial waist size in mm
    R0 = 1.0e30  # Initial curvature radius
    Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
    L1 = 180  # Distance in mm
    z = np.linspace(0, 300, 1000)  # Array of distances in mm
    s = 180 # initial waist position
    # User input for beam quality index
    Mp2 = 1.5
    assert np.allclose(gaussian_beam_through_lens(lambda_, w0, R0, Mf1, z,Mp2,L1,s), target)

    target = targets[2]
    lambda_ = 1.064e-3  # Wavelength in mm
    w0 = 0.2  # Initial waist size in mm
    R0 = 1.0e30  # Initial curvature radius
    Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
    L1 = 180  # Distance in mm
    z = np.linspace(0, 300, 1000)  # Array of distances in mm
    s = 100 # initial waist position
    # User input for beam quality index
    Mp2 = 1.5
    assert np.allclose(gaussian_beam_through_lens(lambda_, w0, R0, Mf1, z,Mp2,L1,s), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e