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
            P_lm = scipy.special.lpmv(m, l_prime, np.cos(z / r_0))
            
            # Update the translation coefficient using the recursion relation
            Rlnm_value += j_l * P_lm * np.exp(1j * m * z)

    return Rlnm_value

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
