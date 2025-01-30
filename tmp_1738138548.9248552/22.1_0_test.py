from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy



# Background: In the context of translating a beam in the z-direction, the translation coefficients 
# are used to express the reexpansion of the beam in terms of spherical harmonics. The translation 
# coefficient (R|R)_{ln}^{m}(r_0) is a complex number that relates the original and translated 
# spherical harmonic expansions. The recursion method is a numerical technique used to efficiently 
# compute these coefficients, especially when dealing with large series expansions. The translation 
# distance z affects the argument of the spherical Bessel functions, which are central to the 
# computation of these coefficients. The wavevector k is related to the wavelength of the beam and 
# influences the oscillatory behavior of the Bessel functions.



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
    translation_coefficient = 0.0 + 0.0j

    # Calculate the argument for the spherical Bessel functions
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

    # Normalize the translation coefficient
    translation_coefficient *= (4 * np.pi) / (2 * l + 1)

    return translation_coefficient


try:
    targets = process_hdf5_to_tuple('22.1', 3)
    target = targets[0]
    l = 1
    n = 1
    m = 0
    wl = 2 * np.pi
    k = 2 * np.pi / wl
    z = 0.75
    N_t = 8
    assert np.allclose(Rlnm(l, n, m, k, z, N_t), target)

    target = targets[1]
    l = 2
    n = 1
    m = 0
    wl = 2 * np.pi
    k = 2 * np.pi / wl
    z = 0.75
    N_t = 8
    assert np.allclose(Rlnm(l, n, m, k, z, N_t), target)

    target = targets[2]
    l = 3
    n = 2
    m = 1
    wl = 2 * np.pi
    k = 2 * np.pi / wl
    z = 0.75
    N_t = 8
    assert np.allclose(Rlnm(l, n, m, k, z, N_t), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e