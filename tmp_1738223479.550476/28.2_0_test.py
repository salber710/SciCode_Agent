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



# Background: Gaussian beams can be characterized by their beam waist and radius of curvature. 
# When a Gaussian beam propagates through an optical system, its properties can be described 
# using the ABCD matrix method. This method uses a ray transfer matrix to determine how the 
# beam's complex beam parameter changes as it passes through optical elements like lenses. 
# The complex beam parameter q is given by 1/q = 1/R - i * (wavelength / (pi * w^2)), where 
# R is the radius of curvature and w is the beam waist size. The ABCD matrix describes the 
# transformation of q through an optical system: q_out = (A*q_in + B) / (C*q_in + D). 
# After propagation, the waist size at a distance z can be found from the imaginary part of 1/q_out.


def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    '''gaussian_beam_through_lens simulates the propagation of a Gaussian beam through an optical lens system
    and calculates the beam waist size at various distances from the lens.
    Input
    - wavelength: float, The wavelength of the light (in meters).
    - w0: float, The waist radius of the Gaussian beam before entering the optical system (in meters).
    - R0: float, The radius of curvature of the beam's wavefront at the input plane (in meters).
    - Mf1: The ABCD matrix representing the first lens or optical element in the system, 2D numpy array with shape (2,2)
    - z: A 1d numpy array of float distances (in meters) from the lens where the waist size is to be calculated.
    - Mp2: float, A scaling factor used in the initial complex beam parameter calculation.
    - L1: float, The distance (in meters) from the first lens or optical element to the second one in the optical system.
    - s: float, the distance (in meters) from the source (or initial beam waist) to the first optical element.
    Output
    wz: waist over z, a 1d array of float, same shape of z
    '''

    # Calculate the initial complex beam parameter q0
    q0 = R0 - 1j * (wavelength * Mp2) / (np.pi * w0**2)

    # Calculate the complex beam parameter after the first lens using the ABCD matrix Mf1
    A, B, C, D = Mf1[0, 0], Mf1[0, 1], Mf1[1, 0], Mf1[1, 1]
    q1 = (A * q0 + B) / (C * q0 + D)

    # Calculate the complex beam parameter after propagation over distance L1
    q2 = q1 + L1

    # Initialize array for the beam waist sizes at different z positions
    wz = np.zeros_like(z)

    # Calculate the beam waist size at each z position
    for i, z_pos in enumerate(z):
        qz = q2 + z_pos
        wz[i] = np.sqrt(-wavelength / (np.pi * np.imag(1/qz)))

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