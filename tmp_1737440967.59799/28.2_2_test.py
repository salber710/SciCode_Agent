import numpy as np
from scipy.integrate import simps

# Background: 
# The Gaussian beam is a fundamental mode of light propagation in optical systems, characterized by its bell-shaped intensity profile. 
# The beam waist (w0) is the location along the beam where the cross-sectional area is minimal, and from there it diverges as it propagates.
# The Fourier transform technique is used to analyze the beam in the frequency domain, which is useful for understanding how the beam 
# evolves over distance. The propagation of the Gaussian beam can be described using the paraxial approximation, and its evolution over 
# a distance z can be computed by multiplying the Fourier transform of the initial field distribution by a phase factor and then 
# transforming back to the spatial domain. The phase factor is derived from the propagation term in the Helmholtz equation.



def propagate_gaussian_beam(N, Ld,


# Background: 
# The propagation of a Gaussian beam through optical systems, such as lenses, can be effectively analyzed using the ABCD matrix method.
# This approach employs the complex beam parameter q, which describes the Gaussian beam in terms of its waist size and curvature.
# The ABCD matrix represents the linear transformation of the beam through optical elements.
# For a Gaussian beam, the complex beam parameter q is given by:
#   q = z - iz_R, where z is the position along the propagation axis and z_R is the Rayleigh range.
# The Rayleigh range is defined as z_R = π * w0^2 / λ, where w0 is the beam waist at the beam waist position.
# When a beam passes through an optical element characterized by an ABCD matrix, the transformed complex beam parameter q' is:
#   q' = (A*q + B) / (C*q + D)
# The waist size w at a distance z can be determined from the imaginary part of the transformed complex beam parameter q':
#   w(z) = sqrt((λ / π) * Im(1/q')).
# This function calculates the beam waist size at various distances from the lens using the given parameters and ABCD matrices.



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
    
    # Calculate the Rayleigh range
    z_R = np.pi * w0**2 / wavelength
    
    # Initial complex beam parameter q
    q_0 = 1 / (1/R0 - 1j * wavelength / (np.pi * w0**2))
    
    # Propagate to the first lens
    s_matrix = np.array([[1, s], [0, 1]])
    q_at_lens = (s_matrix[0, 0] * q_0 + s_matrix[0, 1]) / (s_matrix[1, 0] * q_0 + s_matrix[1, 1])
    
    # Through the first lens
    q_after_lens = (Mf1[0, 0] * q_at_lens + Mf1[0, 1]) / (Mf1[1, 0] * q_at_lens + Mf1[1, 1])
    
    # Propagate to each point in z
    wz = np.zeros_like(z)
    for i, zi in enumerate(z):
        L_matrix = np.array([[1, zi + L1], [0, 1]])
        q_at_z = (L_matrix[0, 0] * q_after_lens + L_matrix[0, 1]) / (L_matrix[1, 0] * q_after_lens + L_matrix[1, 1])
        wz[i] = np.sqrt((wavelength / np.pi) * np.abs(1 / np.imag(q_at_z)))
    
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
