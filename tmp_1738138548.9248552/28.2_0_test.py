from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps


def propagate_gaussian_beam(N, Ld, w0, z, L):
    # Create spatial grid
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # Initial Gaussian beam field distribution
    r2 = X**2 + Y**2
    initial_field = np.exp(-r2 / (w0**2))
    
    # Compute the Fourier transform of the initial field
    initial_field_ft = np.fft.fftshift(np.fft.fft2(initial_field))
    
    # Compute frequency coordinates
    dx = x[1] - x[0]
    df = 1 / (N * dx)
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    FX, FY = np.meshgrid(fx, fy)
    
    # Wavenumber
    k = 2 * np.pi / Ld
    
    # Compute the transfer function for Gaussian beam propagation in the Fourier domain
    H = np.exp(-1j * k * z * np.sqrt(1 - (Ld * FX)**2 - (Ld * FY)**2))
    
    # Apply the transfer function to the Fourier transform of the initial field
    propagated_field_ft = initial_field_ft * H
    
    # Compute the inverse Fourier transform to get the propagated field in the spatial domain
    propagated_field = np.fft.ifft2(np.fft.ifftshift(propagated_field_ft))
    
    # Return the absolute values of the initial and propagated fields
    return np.abs(initial_field), np.abs(propagated_field)



# Background: In optics, the ABCD matrix method is used to describe the transformation of Gaussian beams through optical systems. 
# The complex beam parameter q is used to describe the Gaussian beam, where q = z + i*z_R, with z being the distance along the 
# propagation axis and z_R the Rayleigh range. The transformation of the beam through an optical system is given by the ABCD matrix:
# q' = (A*q + B) / (C*q + D). The beam waist w(z) at a distance z can be calculated from the complex beam parameter q using the 
# relation w(z) = sqrt(-wavelength / (pi * Im(1/q))), where Im denotes the imaginary part. The ABCD matrix for a lens is given by:
# [A B; C D] = [1 0; -1/f 1], where f is the focal length of the lens. The initial complex beam parameter q0 is calculated using 
# the initial waist w0 and the radius of curvature R0 as q0 = R0 + i*(pi*w0^2/wavelength).


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
    z_R = np.pi * w0**2 / wavelength
    q0 = R0 + 1j * z_R

    # Propagate the beam to the first lens
    A1, B1, C1, D1 = 1, s, 0, 1  # Free space propagation matrix
    q1 = (A1 * q0 + B1) / (C1 * q0 + D1)

    # Apply the lens ABCD matrix Mf1
    A2, B2, C2, D2 = Mf1[0, 0], Mf1[0, 1], Mf1[1, 0], Mf1[1, 1]
    q2 = (A2 * q1 + B2) / (C2 * q1 + D2)

    # Propagate the beam to the second lens or to the observation plane
    A3, B3, C3, D3 = 1, L1, 0, 1  # Free space propagation matrix
    q3 = (A3 * q2 + B3) / (C3 * q2 + D3)

    # Calculate the beam waist at each distance z
    wz = np.zeros_like(z)
    for i, zi in enumerate(z):
        A4, B4, C4, D4 = 1, zi, 0, 1  # Free space propagation matrix
        qz = (A4 * q3 + B4) / (C4 * q3 + D4)
        wz[i] = np.sqrt(-wavelength / (np.pi * np.imag(1 / qz)))

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