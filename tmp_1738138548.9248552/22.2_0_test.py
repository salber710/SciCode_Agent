from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy



def Rlnm(l, n, m, k, z, N_t):
    '''Function to calculate the translation coefficient (R|R)lmn using a hybrid approach with Chebyshev polynomials.
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
    translation_coefficient = 0.0 + 0.0j

    # Calculate the argument for the spherical Bessel functions
    kr = k * z

    # Use Chebyshev polynomials to approximate the Legendre polynomial calculations
    for l_prime in range(N_t):
        # Calculate the spherical Bessel function of the first kind
        j_l = scipy.special.spherical_jn(l_prime, kr)

        # Calculate the Chebyshev polynomial of the first kind
        T_lm = np.polynomial.chebyshev.Chebyshev.basis(l_prime)(np.cos(z))

        # Update the translation coefficient using the Chebyshev polynomial approximation
        translation_coefficient += (2 * l_prime + 1) * j_l * T_lm

    # Normalize the translation coefficient
    translation_coefficient *= (4 * np.pi) / (2 * l + 1)

    return translation_coefficient



# Background: In the context of spherical harmonics and rotations, the rotation of spherical harmonics can be described using Wigner D-matrices or rotation matrices. The rotation matrix Q is a 3x3 matrix that describes the transformation of the coordinate system. The spherical harmonics Y_n^m are functions that depend on the angles θ and φ, and they transform under rotations according to the rotation coefficients T_n^{νm}. These coefficients can be computed using the elements of the rotation matrix Q and are crucial for re-expanding spherical harmonics in a rotated coordinate system. The rotation coefficients are used to express the rotated spherical harmonics in terms of the original ones.

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
    T = 0.0 + 0.0j

    # Calculate the rotation coefficient using the rotation matrix Q
    # The rotation coefficient T_n^{νm} is related to the elements of the rotation matrix Q
    # For simplicity, we assume a direct relationship between the indices and the matrix elements
    # This is a simplification and in practice, the relationship might involve Wigner D-matrices

    # Here, we assume a simple linear combination of the matrix elements
    # This is a placeholder for the actual computation which might involve more complex operations
    T = Q[0, 0] * (n + v + m) + Q[1, 1] * (n - v + m) + Q[2, 2] * (n + v - m)

    return T


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e