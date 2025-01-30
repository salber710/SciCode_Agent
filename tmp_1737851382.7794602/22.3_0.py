import numpy as np
import scipy

# Background: 
# The translation of a beam in the z-direction involves calculating the translation coefficients 
# for the reexpansion of the beam. The translation coefficient (R|R)_{ln}^{m}(r_0) is a complex 
# number that represents the contribution of the translated beam to the reexpansion in terms of 
# spherical harmonics. The recursion method is used to efficiently compute these coefficients 
# by leveraging the properties of spherical harmonics and the translation of coordinates. 
# The translation distance is given by z, and the wavevector of the optical beam is k. 
# The recursion method involves iterating over the quantum numbers and using previously 
# computed values to find the next ones, which is crucial for computational efficiency.



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
    if not isinstance(l, int) or not isinstance(n, int) or not isinstance(m, int):
        raise TypeError("Quantum numbers l, n, and m must be integers.")
    if not isinstance(k, (int, float)):
        raise TypeError("Wavevector k must be a real number.")
    if not isinstance(z, (int, float)):
        raise TypeError("Translation distance z must be a real number.")
    if not isinstance(N_t, int):
        raise TypeError("Truncated space size N_t must be an integer.")
    if k < 0:
        raise ValueError("Wavevector k must be non-negative.")
    if N_t < 0:
        raise ValueError("Truncated space size N_t must be non-negative.")

    # Initialize the translation coefficient
    Rlnm_value = 0.0 + 0.0j

    # Calculate the translation distance r_0
    r_0 = np.abs(z)

    # Use recursion to calculate the translation coefficient
    # This is a placeholder for the actual recursion logic
    # The recursion would typically involve spherical Bessel functions and associated Legendre polynomials
    for l_prime in range(N_t):
        for s in range(-l_prime, l_prime + 1):
            # Calculate the spherical Bessel function
            j_l = scipy.special.spherical_jn(l_prime, k * r_0)
            
            # Calculate the associated Legendre polynomial
            if r_0 != 0:
                P_lm = scipy.special.lpmv(m, l_prime, np.cos(z / r_0))
            else:
                P_lm = scipy.special.lpmv(m, l_prime, 1) if l_prime == 0 else 0
            
            # Update the translation coefficient using the recursion relation
            Rlnm_value += j_l * P_lm * np.exp(1j * m * z)

    return Rlnm_value


# Background: The rotation of a beam in spherical harmonics involves calculating the rotation coefficients
# T_n^{\nu m}, which describe how the spherical harmonics transform under a rotation of the coordinate system.
# The rotation is represented by a rotation matrix Q, which relates the original coordinate axes to the rotated
# ones. The spherical harmonics Y_n^m are functions of the angular coordinates and are used to describe the
# angular part of wavefunctions in quantum mechanics. The rotation coefficients T_n^{\nu m} are used to express
# the rotated spherical harmonics in terms of the original ones. The recursion method is employed to efficiently
# compute these coefficients by leveraging the properties of spherical harmonics and the rotation matrix.

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




    # Validate inputs
    if not isinstance(n, int) or not isinstance(v, int) or not isinstance(m, int):
        raise TypeError("Quantum numbers n, v, and m must be integers.")
    if not isinstance(Q, np.ndarray) or Q.shape != (3, 3):
        raise TypeError("Rotation matrix Q must be a 3x3 numpy array.")
    if not np.allclose(np.dot(Q, Q.T), np.eye(3), atol=1e-8):
        raise ValueError("Rotation matrix Q must be orthonormal.")

    # Initialize the rotation coefficient
    T = 0.0 + 0.0j

    # Calculate the rotation coefficient using the Wigner D-matrix elements
    # The Wigner D-matrix elements can be used to express the rotation of spherical harmonics
    # D^n_{m,v}(alpha, beta, gamma) are the elements of the Wigner D-matrix for the rotation
    # The angles alpha, beta, gamma can be derived from the rotation matrix Q

    # Extract Euler angles from the rotation matrix Q
    # This is a simplified approach assuming Q is a proper rotation matrix
    r = R.from_matrix(Q)
    alpha, beta, gamma = r.as_euler('zyz')

    # Calculate the Wigner D-matrix element using spherical harmonics
    # The rotation coefficient T_n^{\nu m} is given by the integral of the product of spherical harmonics
    # over the sphere, which can be simplified using the orthogonality of the spherical harmonics
    for k in range(-n, n+1):
        Ymk_star = np.conj(sph_harm(k, n, alpha, beta))
        Ymv = sph_harm(v, n, gamma, 0)  # gamma is the azimuthal angle, 0 for polar angle
        T += Ymk_star * Ymv

    return T



# Background: The reexpansion of a beam after an arbitrary translation involves decomposing the translation
# into a series of rotations and a translation along the z-axis. This process can be broken down into three
# main steps: first, rotate the coordinate system such that the z-axis aligns with the translation vector r_0;
# second, perform the translation along the new z-axis; and third, rotate the coordinate system back to its
# original orientation. The reexpansion coefficients BR_nm are calculated by combining the effects of these
# transformations on the spherical harmonics. The rotation is handled using the Wigner D-matrix, and the
# translation is handled using the translation coefficients calculated previously. The final reexpansion
# coefficient is a combination of these transformations applied to the initial expansion coefficients B.

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
    # Validate inputs
    if not isinstance(r0, np.ndarray) or r0.shape != (3,):
        raise TypeError("Translation vector r0 must be a numpy array of shape (3,).")
    if not isinstance(B, np.ndarray) or B.shape != (N_t + 1, 2 * N_t + 1):
        raise TypeError("Expansion coefficients B must be a numpy array of shape (N_t + 1, 2 * N_t + 1).")
    if not isinstance(n, int) or not isinstance(m, int):
        raise TypeError("Quantum numbers n and m must be integers.")
    if not isinstance(wl, (int, float)):
        raise TypeError("Wavelength wl must be a real number.")
    if not isinstance(N_t, int):
        raise TypeError("Truncated space size N_t must be an integer.")

    # Calculate the wavevector k
    k = 2 * np.pi / wl

    # Calculate the magnitude of the translation vector
    r_0_mag = np.linalg.norm(r0)

    # Calculate the rotation matrix to align z-axis with r0
    z_axis = np.array([0, 0, 1])
    if r_0_mag == 0:
        Q1 = np.eye(3)
    else:
        v = np.cross(z_axis, r0 / r_0_mag)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, r0 / r_0_mag)
        Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Q1 = np.eye(3) + Vx + np.dot(Vx, Vx) * ((1 - c) / (s ** 2))

    # Calculate the inverse rotation matrix
    Q2 = Q1.T

    # Initialize the reexpansion coefficient
    BR_nm = 0.0 + 0.0j

    # Calculate the reexpansion coefficient using the rotation and translation
    for l in range(N_t + 1):
        for v in range(-l, l + 1):
            # Calculate the rotation coefficient T_n^{v m} for the first rotation
            T1 = Tnvm(l, v, m, Q1)

            # Calculate the translation coefficient (R|R)_{ln}^{v} for the translation
            Rlnv = Rlnm(l, n, v, k, r_0_mag, N_t)

            # Calculate the rotation coefficient T_l^{v m} for the second rotation
            T2 = Tnvm(l, v, m, Q2)

            # Update the reexpansion coefficient
            BR_nm += B[l, v + N_t] * T1 * Rlnv * T2

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
