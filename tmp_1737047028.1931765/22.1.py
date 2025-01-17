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


from scicode.parse.parse import process_hdf5_to_tuple

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
