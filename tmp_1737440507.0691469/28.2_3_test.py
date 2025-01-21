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
# The propagation of a Gaussian beam through an optical system can be described using the ABCD matrix method. The ABCD matrix 
# formalism is used in optics to analyze the behavior of light rays passing through an optical system. For a Gaussian beam, the 
# complex beam parameter q can be defined as 1/q = 1/R - i * (wavelength/pi * w^2), where R is the radius of curvature of the 
# wavefront, w is the beam waist, and i is the imaginary unit. As the beam propagates through an optical element described by 
# an ABCD matrix, the beam parameter transforms according to q' = (A*q + B)/(C*q + D). The beam waist size at any point can 
# be determined from the transformed beam parameter q'. This method allows us to calculate how the Gaussian beam's waist size 
# evolves as it propagates through lenses and other optical components.


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
    q0 = (R0 - 1j * (wavelength / (np.pi * w0**2))) * Mp2

    # Propagation from source to first lens
    A_s = 1
    B_s = s
    C_s = 0
    D_s = 1
    q1 = (A_s * q0 + B_s) / (C_s * q0 + D_s)

    # Apply the ABCD matrix for the lens
    A_lens = Mf1[0, 0]
    B_lens = Mf1[0, 1]
    C_lens = Mf1[1, 0]
    D_lens = Mf1[1, 1]
    q2 = (A_lens * q1 + B_lens) / (C_lens * q1 + D_lens)

    # Propagation from first lens to second lens (or to the point of interest)
    A_L1 = 1
    B_L1 = L1
    C_L1 = 0
    D_L1 = 1
    q3 = (A_L1 * q2 + B_L1) / (C_L1 * q2 + D_L1)

    # Calculate the beam waist at each distance in z
    wz = np.zeros_like(z)
    for i, zi in enumerate(z):
        A_z = 1
        B_z = zi
        C_z = 0
        D_z = 1
        qz = (A_z * q3 + B_z) / (C_z * q3 + D_z)
        wz[i] = np.sqrt(-wavelength / (np.pi * np.imag(1 / qz)))

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
