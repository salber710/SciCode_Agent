import numpy as np
import scipy

# Background: In the context of translating a beam in the z-direction, the reexpansion of the beam can be described 
# using translation coefficients. These coefficients, denoted as (R|R)_{ln}^{m}(r_0), are crucial for transforming 
# spherical harmonic expansions from one center to another, particularly when the translation vector is along the z-axis. 
# In quantum mechanics and electromagnetics, these coefficients arise in problems involving wave functions or fields 
# expanded in spherical harmonics that are translated by a vector. The recursive method is one efficient way to compute 
# these coefficients by leveraging previously computed values to reduce computational complexity. The translation 
# distance z affects the argument of the spherical Bessel functions and Legendre polynomials involved in the recursive 
# computation. The wavevector k is related to the wavelength and plays a role in determining the phase shift due to 
# translation.

def Rlnm(l, n, m, k, z, N_t):
    '''Function to calculate the translation coefficient (R|R)lmn.
    Input
    l : int
        The principal quantum number for the expansion.
    n : int
        The principal quantum number for the reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    k : float
        Wavevector of the optical beam.
    z : float
        The translation distance along z direction
    N_t : int
        Truncated space size.
    Output
    (R|R)lmn : complex
        The translation coefficient (R|R)lmn.
    '''



    # Initialize the translation coefficient array
    R_ln_m = np.zeros((N_t + 1, N_t + 1), dtype=complex)
    
    # Compute the translation coefficient using a recursion method
    for l_prime in range(N_t + 1):
        for s in range(-l_prime, l_prime + 1):
            # Calculate the argument for the spherical Bessel function
            kr = k * z
            
            # Compute the spherical Bessel function
            j_lprime = scipy.special.spherical_jn(l_prime, kr)
            
            # Compute the associated Legendre polynomial factor
            legendre_factor = scipy.special.lpmv(m, l_prime, np.cos(z))
            
            # Calculate the translation coefficient (R|R)_{ln}^{m}
            R_ln_m[l_prime, s] = j_lprime * legendre_factor * (1j)**(l - l_prime)
    
    # Sum up all contributions for the given l, n, m
    translation_coefficient = np.sum(R_ln_m)
    
    return translation_coefficient


# Background: In the context of rotating spherical harmonics, the rotation coefficients T_n^{\nu m} play a 
# crucial role in transforming spherical harmonics from one coordinate system to another. These coefficients 
# arise from the unitary transformation of spherical harmonics under rotation, described by the rotation matrix Q.
# The spherical harmonics Y_n^m are functions of the angular coordinates, and their transformation under 
# rotation is governed by the rotation group SO(3). The calculation of the rotation coefficients involves 
# evaluating the Wigner D-matrix elements, which represent the matrix form of the rotation operator in 
# the space of spherical harmonics. The recursion method can be used to efficiently compute these coefficients 
# by leveraging the properties of Wigner D-matrices and previously computed terms.

def Tnvm(n, v, m, Q):
    '''Function to calculate the rotation coefficient Tnvm.
    Input
    n : int
        The principal quantum number for both expansion and reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    v : int
        The magnetic quantum number for the expansion.
    Q : matrix of shape(3, 3)
        The rotation matrix.
    Output
    T : complex
        The rotation coefficient Tnvm.
    '''
    # Initialize the rotation coefficient as a complex number
    T = complex(0, 0)
    
    # Calculate the Wigner D-matrix elements using the rotation matrix Q
    # Wigner D-matrix elements are related to the Euler angles which can be derived from the rotation matrix Q
    # For simplicity, assume Q is decomposed into Euler angles (alpha, beta, gamma)
    # Here we directly use the scipy.special.wigner functions to compute D-matrix elements
    
    # Extract Euler angles from Q (this would require a proper decomposition function)
    # For now, assume we have the angles (alpha, beta, gamma)
    # This step must be implemented properly with a function to extract Euler angles from Q
    
    # Placeholder angles (to be replaced with actual extraction logic)
    alpha, beta, gamma = 0.0, 0.0, 0.0
    
    # Compute the Wigner D-matrix element
    D_nvm = scipy.special.wigner_d(n, v, m, beta, alpha, gamma)
    
    # Wigner D-matrix provides the rotation coefficients directly
    T = D_nvm
    
    return T



# Background: The task is to calculate the reexpansion coefficient BR_nm for a translated and rotated beam.
# When translating a beam arbitrarily in space, it can be decomposed into a rotation to align the z-axis with 
# the translation direction, followed by a translation along the new z-axis, and finally a rotation to return 
# to the original orientation. This process involves calculating translation coefficients for the z-direction 
# and rotation coefficients based on spherical harmonics transformations. The reexpansion coefficients are then 
# determined by combining these transformations, leveraging both translation and rotation coefficients. These 
# coefficients are critical in optics for modeling how beams change when they are moved and rotated, using 
# spherical harmonics for angular dependence and the wavevector for phase changes.



def compute_BRnm(r0, B, n, m, wl, N_t):
    '''Function to calculate the reexpansion coefficient BR_nm.
    Input
    r_0 : array
        Translation vector.
    B : matrix of shape(N_t + 1, 2 * N_t + 1)
        Expansion coefficients of the elementary regular solutions.
    n : int
        The principal quantum number for the reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    wl : float
        Wavelength of the optical beam.
    N_t : int
        Truncated space size.
    Output
    BR_nm : complex
        Reexpansion coefficient BR_nm of the elementary regular solutions.
    '''

    # Calculate the magnitude of the translation vector r0
    r0_magnitude = np.linalg.norm(r0)
    k = 2 * np.pi / wl  # Wavevector based on the wavelength

    # Calculate the Euler angles for the rotation aligning z-axis with r0
    theta = np.arccos(r0[2] / r0_magnitude)  # Angle between z-axis and r0
    phi = np.arctan2(r0[1], r0[0])  # Azimuthal angle in xy-plane

    # Initialize the reexpansion coefficient
    BRnm = complex(0, 0)

    # Compute the rotation coefficients (Wigner D-matrix elements)
    for v in range(-n, n + 1):
        D_nvm = scipy.special.wigner_d(n, v, m, theta, phi, 0)  # Rotation from (r0 direction to z-axis)

        # Calculate the translation coefficient for z translation
        R_ln_m = np.zeros((N_t + 1, 2 * N_t + 1), dtype=complex)
        for l in range(N_t + 1):
            for s in range(-l, l + 1):
                kr = k * r0_magnitude
                j_l = scipy.special.spherical_jn(l, kr)
                legendre_factor = scipy.special.lpmv(m, l, np.cos(r0_magnitude))
                R_ln_m[l, s] = j_l * legendre_factor * (1j)**(n - l)

        # Sum up contributions for the given n, m
        for l in range(N_t + 1):
            for s in range(-l, l + 1):
                BRnm += B[l, s] * R_ln_m[l, s] * D_nvm

    return BRnm

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('22.3', 4)
target = targets[0]

r0 = np.array([0.5, 0, 0])
N_t = 5
B = np.zeros((N_t + 1, 2 * N_t + 1))
B[1, N_t] = 1
wl = 2 * np.pi
n = 2
m = 1
assert np.allclose(compute_BRnm(r0, B, n, m, wl, N_t), target)
target = targets[1]

r0 = np.array([0.5, 0.5, 0])
N_t = 5
B = np.zeros((N_t + 1, 2 * N_t + 1))
B[1, N_t] = 1
wl = 2 * np.pi
n = 2
m = 1
assert np.allclose(compute_BRnm(r0, B, n, m, wl, N_t), target)
target = targets[2]

r0 = np.array([0.5, 1, 0])
N_t = 5
B = np.zeros((N_t + 1, 2 * N_t + 1))
B[1, N_t] = 1
wl = 2 * np.pi
n = 2
m = 1
assert np.allclose(compute_BRnm(r0, B, n, m, wl, N_t), target)
target = targets[3]

r0 = np.array([0, 0.5, 0])
N_t = 5
B = np.zeros((N_t + 1, 2 * N_t + 1))
B[1, N_t + 1] = 1
wl = 2 * np.pi
n = 2
m = 2
assert (compute_BRnm(r0, B, n, m, wl, N_t) == 0) == target
