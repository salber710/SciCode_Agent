import numpy as np
import scipy



# Background: 
# In electromagnetic theory, translating a beam along the z-axis involves calculating translation coefficients that relate the 
# original and translated beam's spherical wave expansions. The translation coefficient (R|R)_{ln}^{m}(r_0) is crucial for
# understanding how each spherical wave component is affected by the translation. These coefficients are derived using recursion
# relations, which can efficiently compute them for large systems. The recursion method leverages the recurrence relations
# of spherical Bessel functions and associated Legendre polynomials. The translation distance r_0 is the magnitude of the 
# translation vector in the z-direction, and the wavevector k relates to the beam's characteristics. The truncation parameter N_t 
# determines the size of the truncated expansion space.



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
    
    # Calculate the translation distance in terms of the wavevector
    r0 = k * z
    
    # Initialize the translation coefficient
    translation_coefficient = 0
    
    # Recursion method to calculate the translation coefficient
    # Iterate over possible values of l, s for the series
    for l_prime in range(N_t):
        for s in range(-l_prime, l_prime + 1):
            # Calculate the current term using spherical Bessel functions
            # and associated Legendre polynomials
            jl_r0 = sp.spherical_jn(l_prime, r0)
            pl_cos_theta = sp.lpmv(m, l_prime, np.cos(z))
            
            # Accumulate the term in the translation coefficient
            current_term = (2*l_prime + 1) * jl_r0 * pl_cos_theta
            translation_coefficient += current_term
    
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
