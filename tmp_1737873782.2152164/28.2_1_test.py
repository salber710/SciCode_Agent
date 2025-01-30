from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps


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
    Ouput
    Gau:     a 2D array with dimensions (N+1, N+1) representing the absolute value of the beam's amplitude distribution before propagation.
    Gau_Pro: a 2D array with dimensions (N+1, N+1) representing the absolute value of the beam's amplitude distribution after propagation.
    '''
    
    # Define the spatial grid
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the Rayleigh range
    z_R = np.pi * w0**2 / Ld
    
    # Calculate the waist size at distance z
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)
    
    # Calculate the beam amplitude distribution before propagation
    Gau = np.exp(-(X**2 + Y**2) / w0**2)
    
    # Transform into Fourier domain
    Gau_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Gau)))
    
    # Frequency space coordinates
    fx = np.fft.fftfreq(N, d=L/N)
    fy = np.fft.fftfreq(N, d=L/N)
    FX, FY = np.meshgrid(fx, fy)
    
    # Calculate the phase shift due to propagation in the Fourier domain
    H = np.exp(-1j * np.pi * Ld * z * (FX**2 + FY**2))
    
    # Apply the phase shift
    Gau_Pro_ft = Gau_ft * H
    
    # Inverse transform to get back to spatial domain
    Gau_Pro = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Gau_Pro_ft))))
    
    return Gau, Gau_Pro




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

    # Initialize the output array for beam waist size at each distance z
    wz = np.zeros_like(z)

    # Calculate initial Rayleigh range
    zR0 = np.pi * w0**2 / wavelength

    # Initial complex beam parameter q0
    q0 = 1 / (1/R0 - 1j * wavelength / (np.pi * w0**2))

    for i, zi in enumerate(z):
        # ABCD matrix for free space propagation from source to first lens
        M_free_space = np.array([[1, s],
                                 [0, 1]])

        # Calculate the total ABCD matrix for propagation to distance zi
        M_total = Mf1 @ M_free_space

        # Propagation through free space to distance zi
        q1 = (M_total[0, 0] * q0 + M_total[0, 1]) / (M_total[1, 0] * q0 + M_total[1, 1])

        # Calculate the new waist size at distance zi using the complex beam parameter
        wz[i] = np.sqrt(-wavelength / (np.pi * np.imag(1/q1)))

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