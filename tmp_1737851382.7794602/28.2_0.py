import numpy as np
from scipy.integrate import simps

# Background: 
# A Gaussian beam is a type of electromagnetic wave whose electric field amplitude distribution follows a Gaussian function. 
# The beam is characterized by its waist (w0), which is the location where the beam is narrowest. 
# As the beam propagates, its width increases, and its amplitude decreases. 
# The propagation of a Gaussian beam can be analyzed using Fourier optics, where the beam is transformed into the frequency domain, 
# manipulated, and then transformed back into the spatial domain. 
# The Fourier transform allows us to analyze how the beam's spatial frequency components evolve over distance. 
# The propagation in the Fourier domain is often simplified using the angular spectrum method, which involves multiplying by a phase factor.
# The intensity distribution of the beam is the square of the absolute value of the field distribution.


def propagate_gaussian_beam(N, Ld, w0, z, L):
    '''Propagate a Gaussian beam and calculate its intensity distribution before and after propagation.
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
    '''

    # Create a grid of spatial coordinates
    x = np.linspace(-L/2, L/2, N+1)
    y = np.linspace(-L/2, L/2, N+1)
    X, Y = np.meshgrid(x, y)

    # Calculate the initial Gaussian beam field distribution
    r_squared = X**2 + Y**2
    Gau = np.exp(-r_squared / w0**2) if w0 != 0 else np.zeros((N+1, N+1))

    # Fourier transform of the initial field
    Gau_ft = np.fft.fftshift(np.fft.fft2(Gau))

    # Frequency coordinates
    fx = np.fft.fftfreq(N+1, d=L/(N+1))
    fy = np.fft.fftfreq(N+1, d=L/(N+1))
    FX, FY = np.meshgrid(fx, fy)

    # Calculate the transfer function for propagation
    k = 2 * np.pi / Ld if Ld != 0 else 0  # Handle zero or negative wavelength
    # Avoiding negative values under the square root
    under_sqrt = np.maximum(0, 1 - (Ld * FX)**2 - (Ld * FY)**2)
    H = np.exp(-1j * k * z * np.sqrt(under_sqrt))

    # Apply the transfer function in the Fourier domain
    Gau_Pro_ft = Gau_ft * H

    # Inverse Fourier transform to get the propagated field
    Gau_Pro = np.fft.ifft2(np.fft.ifftshift(Gau_Pro_ft))

    # Return the absolute value of the field distributions
    return np.abs(Gau), np.abs(Gau_Pro)



# Background: 
# The propagation of a Gaussian beam through an optical system can be analyzed using the ABCD matrix method. 
# This method uses a 2x2 matrix to describe the effect of optical elements (like lenses) on the beam's parameters.
# The complex beam parameter q is defined as 1/q = 1/R - i * (wavelength / (π * w^2)), where R is the radius of curvature 
# of the beam's wavefront and w is the beam waist. The transformation of the beam parameter through an optical system 
# is given by q' = (A*q + B) / (C*q + D), where A, B, C, and D are elements of the ABCD matrix.
# The waist size w(z) at a distance z from the lens can be calculated from the imaginary part of the complex beam parameter q(z).


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

    # Initial complex beam parameter q0
    q0 = 1 / (1/R0 - 1j * (wavelength / (np.pi * w0**2)))

    # Propagate q0 through the first lens using the ABCD matrix Mf1
    A, B, C, D = Mf1[0, 0], Mf1[0, 1], Mf1[1, 0], Mf1[1, 1]
    q1 = (A * q0 + B) / (C * q0 + D)

    # Propagate q1 over distance L1
    q2 = q1 + L1

    # Calculate the beam waist size at each distance z
    wz = np.zeros_like(z)
    for i, zi in enumerate(z):
        qz = q2 + zi
        wz[i] = np.sqrt(-wavelength / (np.pi * np.imag(1/qz)))

    return wz

from scicode.parse.parse import process_hdf5_to_tuple
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
