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
