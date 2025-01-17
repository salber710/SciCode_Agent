import numpy as np
import scipy

# Background: In the context of translating a beam in the z-direction, the translation coefficients 
# are used to express the reexpansion of the beam in terms of spherical harmonics. The translation 
# coefficient (R|R)_{ln}^{m}(r_0) is calculated using a recursion method. This involves summing over 
# spherical harmonics and using properties of Bessel functions and Legendre polynomials. The recursion 
# method is efficient for computing these coefficients, especially when dealing with large values of 
# l, n, and m. The translation distance z affects the argument of the Bessel functions, which are 
# central to the calculation of these coefficients.



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
    # Initialize the translation coefficient
    translation_coefficient = 0.0

    # Calculate the argument for the spherical Bessel function
    kr = k * z

    # Use recursion to calculate the translation coefficient
    for l_prime in range(N_t):
        for s in range(-l_prime, l_prime + 1):
            # Calculate the spherical Bessel function of the first kind
            j_l = scipy.special.spherical_jn(l_prime, kr)
            
            # Calculate the associated Legendre polynomial
            P_lm = scipy.special.lpmv(m, l_prime, np.cos(z))
            
            # Update the translation coefficient using the recursion relation
            translation_coefficient += (2 * l_prime + 1) * j_l * P_lm

    # Return the calculated translation coefficient
    return translation_coefficient


# Background: In the context of rotating a beam, the rotation coefficients T_n^{\nu m} are used to 
# express the reexpansion of spherical harmonics after a rotation. The rotation is described by a 
# rotation matrix Q, which relates the original coordinate system to the rotated one. The spherical 
# harmonics Y_n^m are functions of the angular coordinates and are used in many areas of physics 
# and engineering, particularly in problems involving angular momentum. The rotation coefficients 
# are calculated using the properties of spherical harmonics and the rotation matrix. The recursion 
# method is efficient for computing these coefficients, especially when dealing with large values 
# of n, m, and v. The rotation matrix Q is a 3x3 matrix that describes the transformation of the 
# coordinate system.

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
    # Initialize the rotation coefficient
    T = 0.0

    # Calculate the Wigner D-matrix elements for the rotation
    # The Wigner D-matrix is used to describe the rotation of spherical harmonics
    # Here, we use scipy's special function to calculate the Wigner D-matrix elements
    for m_prime in range(-n, n + 1):
        # Calculate the Wigner D-matrix element
        D_element = scipy.special.sph_harm(m_prime, n, Q[0, 2], Q[1, 2])  # Using Q to get angles

        # Update the rotation coefficient using the Wigner D-matrix element
        T += D_element * scipy.special.sph_harm(v, n, Q[0, 0], Q[1, 0])

    # Return the calculated rotation coefficient
    return T



# Background: In the context of translating a beam in an arbitrary direction, the translation can be 
# decomposed into a series of rotations and a translation along the z-axis. This involves rotating the 
# coordinate system such that the z-axis aligns with the translation vector r0, performing the translation 
# along this new z-axis, and then rotating the system back to its original orientation. The reexpansion 
# coefficients BR_nm are calculated by considering these transformations and using the previously computed 
# translation and rotation coefficients. The process involves spherical harmonics and the properties of 
# rotation matrices, which are used to express the transformation of the beam in terms of its spherical 
# harmonic components.

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
    # Calculate the magnitude of the translation vector
    r0_magnitude = np.linalg.norm(r0)
    
    # Calculate the wavevector
    k = 2 * np.pi / wl
    
    # Calculate the angles for the rotation to align z-axis with r0
    theta = np.arccos(r0[2] / r0_magnitude)
    phi = np.arctan2(r0[1], r0[0])
    
    # Construct the rotation matrix Q for aligning z-axis with r0
    Q = np.array([
        [np.cos(phi) * np.cos(theta), -np.sin(phi), np.cos(phi) * np.sin(theta)],
        [np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # Initialize the reexpansion coefficient
    BR_nm = 0.0
    
    # Iterate over the expansion coefficients
    for l in range(N_t + 1):
        for v in range(-l, l + 1):
            # Calculate the rotation coefficient T_n^{v m}
            T_nvm = Tnvm(n, v, m, Q)
            
            # Calculate the translation coefficient (R|R)_{ln}^{v}(r0_magnitude)
            Rlnv = Rlnm(l, n, v, k, r0_magnitude, N_t)
            
            # Update the reexpansion coefficient
            BR_nm += B[l, v + N_t] * T_nvm * Rlnv
    
    return BR_nm


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
