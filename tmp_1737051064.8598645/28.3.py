import numpy as np
from scipy.integrate import simps

# Background: 
# A Gaussian beam is a type of electromagnetic wave that has a Gaussian intensity profile. The beam's field distribution can be described by the Gaussian beam equation. 
# When a Gaussian beam propagates through space, its width and phase change according to the distance traveled. The beam's waist (w0) is the location where the beam is narrowest.
# The Fourier transform is a mathematical tool that allows us to analyze the frequency components of a signal. In optics, it is often used to switch between the spatial domain and the frequency domain.
# The propagation of a Gaussian beam can be efficiently calculated in the Fourier domain using the angular spectrum method. This involves taking the Fourier transform of the initial field distribution, applying a phase shift that accounts for propagation, and then transforming back to the spatial domain.
# The phase shift in the Fourier domain is given by the propagation factor exp(-i * k * z), where k is the wave number (2 * pi / wavelength) and z is the propagation distance.



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

    # Define the spatial grid
    x = np.linspace(-L/2, L/2, N+1)
    y = np.linspace(-L/2, L/2, N+1)
    X, Y = np.meshgrid(x, y)

    # Calculate the initial Gaussian beam field distribution
    r_squared = X**2 + Y**2
    Gau = np.exp(-r_squared / w0**2)

    # Fourier transform of the initial field
    Gau_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Gau)))

    # Define the frequency grid
    fx = np.fft.fftfreq(N+1, d=L/(N+1))
    fy = np.fft.fftfreq(N+1, d=L/(N+1))
    FX, FY = np.meshgrid(fx, fy)

    # Calculate the wave number
    k = 2 * np.pi / Ld

    # Calculate the propagation phase factor in the Fourier domain
    H = np.exp(-1j * k * z * np.sqrt(1 - (Ld * FX)**2 - (Ld * FY)**2))

    # Apply the propagation phase factor
    Gau_Pro_ft = Gau_ft * H

    # Inverse Fourier transform to get the propagated field
    Gau_Pro = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Gau_Pro_ft)))

    # Return the absolute value of the field distributions
    return np.abs(Gau), np.abs(Gau_Pro)


# Background: 
# The propagation of a Gaussian beam through an optical system can be analyzed using the ABCD matrix method. 
# This method uses a 2x2 matrix to describe the transformation of the beam's complex beam parameter q as it passes through optical elements.
# The complex beam parameter q is defined as 1/q = 1/R - i * (wavelength / (π * w^2)), where R is the radius of curvature of the beam's wavefront, and w is the beam waist.
# For a Gaussian beam passing through an optical system described by an ABCD matrix, the output complex beam parameter q' is given by q' = (A*q + B) / (C*q + D).
# The beam waist w(z) at a distance z from the lens can be calculated from the imaginary part of the complex beam parameter q(z).
# The ABCD matrix for free space propagation over a distance d is given by [[1, d], [0, 1]].
# The waist size w(z) can be extracted from the imaginary part of the complex beam parameter q(z) using the relation w(z) = sqrt(-wavelength / (π * Im(1/q(z)))).


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

    # Propagation from source to first lens
    M_free_space_s = np.array([[1, s], [0, 1]])
    q1 = (M_free_space_s[0, 0] * q0 + M_free_space_s[0, 1]) / (M_free_space_s[1, 0] * q0 + M_free_space_s[1, 1])

    # Through the first lens
    q2 = (Mf1[0, 0] * q1 + Mf1[0, 1]) / (Mf1[1, 0] * q1 + Mf1[1, 1])

    # Propagation from first lens to second lens
    M_free_space_L1 = np.array([[1, L1], [0, 1]])
    q3 = (M_free_space_L1[0, 0] * q2 + M_free_space_L1[0, 1]) / (M_free_space_L1[1, 0] * q2 + M_free_space_L1[1, 1])

    # Calculate waist size at each distance z
    wz = np.zeros_like(z)
    for i, zi in enumerate(z):
        # Propagation from second lens to distance z[i]
        M_free_space_zi = np.array([[1, zi], [0, 1]])
        qz = (M_free_space_zi[0, 0] * q3 + M_free_space_zi[0, 1]) / (M_free_space_zi[1, 0] * q3 + M_free_space_zi[1, 1])
        wz[i] = np.sqrt(-wavelength / (np.pi * np.imag(1/qz)))

    return wz



# Background: 
# The propagation of a Gaussian beam through free space and optical systems can be analyzed using the ABCD matrix method and Fourier optics.
# The ABCD matrix method allows us to calculate the transformation of the beam's complex beam parameter q as it passes through optical elements.
# The waist size of the beam at any point can be determined from the imaginary part of the complex beam parameter.
# The Fourier optics approach can be used to calculate the intensity distribution of the beam at any plane by considering the beam's field distribution in the Fourier domain.
# The new focus of the beam after passing through a lens can be determined by finding the position where the beam waist is minimized.
# The intensity distribution at the new focus can be calculated by propagating the beam to the focus plane and evaluating the field distribution.

def Gussian_Lens_transmission(N, Ld, z, L, w0, R0, Mf1, Mp2, L1, s):
    '''This function runs guassian beam transmission simlation for free space and lens system. 
    Inputs
    N : int
        The number of sampling points in each dimension (assumes a square grid).
    Ld : float
        Wavelength of the Gaussian beam.
    z:  A 1d numpy array of float distances (in meters) from the lens where the waist size is to be calculated.
    L : float
        Side length of the square area over which the beam is sampled.
    w0 : float
        Initial Waist radius of the Gaussian beam at its narrowest point before incident light.
    R0: float
        The radius of curvature of the beam's wavefront at the input plane (in meters).
    Mf1: 2*2 float matrix
        The ABCD matrix representing the first lens or optical element in the system (2x2 numpy array).
    Mp2: float
        A scaling factor used in the initial complex beam parameter calculation.
    L1: float
        The distance (in meters) from the first lens or optical element to the second one in the optical system.
    s: float
        the distance (in meters) from the source (or initial beam waist) to the first optical element.
    Outputs
    Wz: 1D array with float element; Waist over z axis (light papragation axis)
    focus_depth: float; new focus position through lens  
    Intensity: float 2D array; The intensity distribution at new fcous after the lens
    '''


    # Calculate the waist size over the z axis using the ABCD matrix method
    Wz = gaussian_beam_through_lens(Ld, w0, R0, Mf1, z, Mp2, L1, s)

    # Find the new focus position (minimum waist size)
    focus_depth_index = np.argmin(Wz)
    focus_depth = z[focus_depth_index]

    # Calculate the intensity distribution at the new focus plane
    # Propagate the beam to the focus plane using the Fourier optics method
    _, Intensity = propagate_gaussian_beam(N, Ld, Wz[focus_depth_index], focus_depth, L)

    return Wz, focus_depth, Intensity


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('28.3', 4)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
Ld = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 150  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 0
# User input for beam quality index
Mp2 = 1
N = 800  
L = 16*10**-3
assert cmp_tuple_or_list(Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
Ld= 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 100 # initial waist position
# User input for beam quality index
Mp2 = 1.5
N = 800  
L = 16*10**-3
assert cmp_tuple_or_list(Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
Ld = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 150 # initial waist position
# User input for beam quality index
Mp2 = 1.5
N = 800  
L = 16*10**-3
assert cmp_tuple_or_list(Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s), target)
target = targets[3]

Ld = 1.064e-3  # Wavelength in mm
w0 = 0.2  # Initial waist size in mm
R0 = 1.0e30  # Initial curvature radius
Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
L1 = 180  # Distance in mm
z = np.linspace(0, 300, 1000)  # Array of distances in mm
s = 150 # initial waist position
# User input for beam quality index
Mp2 = 1.5
N = 800  
L = 16*10**-3
Wz, _, _ = Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s)
scalingfactor = max(z)/len(z)
LensWz = Wz[round(L1/scalingfactor)]
BeforeLensWz = Wz[round((L1-5)/scalingfactor)]
AfterLensWz = Wz[round((L1+5)/scalingfactor)]
assert (LensWz>BeforeLensWz, LensWz>AfterLensWz) == target
