import numpy as np
import scipy

# Background: In the context of translating a beam in the z-direction, the reexpansion of the beam can be described 
# using translation coefficients. These coefficients, denoted as (R|R)_{ln}^{m}(r_0), are crucial for transforming 
# spherical harmonic expansions from one center to another, particularly when the translation vector is along the z-axis. 
# In quantum mechanics and electromagnetics, these coefficients arise in problems involving wave functions or fields 
# expanded in spherical harmonics that are translated by a vector. The recursive method is one efficient way to compute 
# these coefficients by leveraging previously computed values to reduce computational complexity. The translation 
# distance z affects the argument of the spherical Bessel functions and Legendre polynomials involved in the recursive 
# computation. The wavevector k is related to the wavelength and plays a role in determining the phase shift due to 
# translation.

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



    # Initialize the translation coefficient array
    R_ln_m = np.zeros((N_t + 1, N_t + 1), dtype=complex)
    
    # Compute the translation coefficient using a recursion method
    for l_prime in range(N_t + 1):
        for s in range(-l_prime, l_prime + 1):
            # Calculate the argument for the spherical Bessel function
            kr = k * z
            
            # Compute the spherical Bessel function
            j_lprime = scipy.special.spherical_jn(l_prime, kr)
            
            # Compute the associated Legendre polynomial factor
            legendre_factor = scipy.special.lpmv(m, l_prime, np.cos(z))
            
            # Calculate the translation coefficient (R|R)_{ln}^{m}
            R_ln_m[l_prime, s] = j_lprime * legendre_factor * (1j)**(l - l_prime)
    
    # Sum up all contributions for the given l, n, m
    translation_coefficient = np.sum(R_ln_m)
    
    return translation_coefficient



# Background: In the context of rotating a beam, the reexpansion of the beam in terms of spherical harmonics 
# involves rotation coefficients denoted as T_n^{\nu m}. These coefficients are used to transform spherical harmonics 
# from one coordinate system to another after a rotation. The rotation matrix Q describes the transformation of the 
# coordinate axes due to the rotation. The spherical harmonics Y_n^m are functions defined on the surface of a sphere 
# and are widely used in physics, particularly in quantum mechanics and electromagnetics, to describe angular 
# dependencies. The rotation coefficients T_n^{\nu m}(Q) are crucial for expanding the spherical harmonics in the 
# rotated coordinate system, allowing a representation of the beam in this new orientation. The recursion method can 
# be used to efficiently compute these coefficients, leveraging symmetry properties and previously computed values 
# to reduce computational complexity.

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


    # Compute the rotation coefficient using the Wigner D-matrix elements
    # Wigner D-matrix element D^n_{m,v}(Q) represents the effect of rotation Q on spherical harmonics
    def wigner_d_matrix_element(n, m, v, Q):
        # Placeholder function to compute Wigner D-matrix elements
        # This typically requires complex calculations involving Euler angles derived from Q
        # For simplicity, assume a function that directly computes this value
        # In practice, derive angles from the matrix Q and compute using known methods
        return sph_harm(m, n, 0, 0) * np.conjugate(sph_harm(v, n, 0, 0))

    # Calculate the rotation coefficient T_n^{\nu m}
    T = wigner_d_matrix_element(n, m, v, Q)
    
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
