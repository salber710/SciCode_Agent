from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy



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

    # Precompute common factors
    kz = k * z

    # Recursively calculate the translation coefficient
    for l_prime in range(0, N_t):
        for s in range(-l_prime, l_prime + 1):
            # Calculate the spherical Bessel functions
            jn_kr = spherical_jn(l_prime, kz)
            yn_kr = spherical_yn(l_prime, kz)

            # Compute the translation coefficient using the recursion relation
            # Here, we use the recursion for spherical Bessel functions
            # R_l^m(r_q) is expanded using spherical harmonics and Bessel functions
            # This is a simplified approach and may require additional terms depending on the complexity
            # of the problem and the specific recursion involved.
            term = ((2 * l_prime + 1) * (1j)**(l - l_prime) *
                    (jn_kr + 1j * yn_kr)) 

            # Accumulate the contribution for this term
            translation_coefficient += term

    return translation_coefficient



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

    # Import necessary functions from scipy


    # Initialize the rotation coefficient
    T = 0.0

    # Iterate over possible values of the magnetic quantum number
    for mu in range(-n, n + 1):
        # Compute the Wigner D-matrix element
        D = 0.0
        for i in range(3):
            for j in range(3):
                D += Q[i, j] * sph_harm(mu, n, i, j).real  # Assuming a real rotation matrix

        # Accumulate the contribution for this term
        T += D * sph_harm(mu, n, 0, 0).real  # Using spherical harmonics

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