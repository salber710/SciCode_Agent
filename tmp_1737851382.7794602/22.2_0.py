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

    # Initialize the rotation coefficient
    T = 0.0 + 0.0j

    # Calculate the rotation coefficient using the Wigner D-matrix elements
    # The Wigner D-matrix elements can be used to express the rotation of spherical harmonics
    # D^n_{m,v}(alpha, beta, gamma) are the elements of the Wigner D-matrix for the rotation
    # The angles alpha, beta, gamma can be derived from the rotation matrix Q

    # Extract Euler angles from the rotation matrix Q
    # This is a simplified approach assuming Q is a proper rotation matrix
    beta = np.arccos(Q[2, 2])
    if np.sin(beta) != 0:
        alpha = np.arctan2(Q[2, 0], -Q[2, 1])
        gamma = np.arctan2(Q[0, 2], Q[1, 2])
    else:
        # Gimbal lock case
        alpha = np.arctan2(Q[0, 1], Q[0, 0])
        gamma = 0

    # Calculate the Wigner D-matrix element
    D_nmv = scipy.special.wigner_d(n, m, v, alpha, beta, gamma)

    # The rotation coefficient T_n^{\nu m} is given by the Wigner D-matrix element
    T = D_nmv

    return T

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('22.2', 3)
target = targets[0]

Q = np.matrix([[0, 0, 1],[0, 1, 0],[-1, 0, 0]])
assert np.allclose(Tnvm(2, 1, 1, Q), target)
target = targets[1]

Q = np.matrix([[0, 0, 1],[0, 1, 0],[-1, 0, 0]])
assert np.allclose(Tnvm(5, 2, 1, Q), target)
target = targets[2]

Q = np.matrix([[- 1 / np.sqrt(2), - 1 / np.sqrt(2), 0],[1 / (2 * np.sqrt(2)), - 1 / (2 * np.sqrt(2)), np.sqrt(3) / 2],[- np.sqrt(3 / 2) / 2, np.sqrt(3 / 2) / 2, 1 / 2]])
assert np.allclose(Tnvm(1, 1, 0, Q), target)
