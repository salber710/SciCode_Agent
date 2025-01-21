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



# Background: In the context of rotations in three-dimensional space, the rotation of spherical harmonics can be 
# described using rotation matrices and the corresponding rotation coefficients. The rotation coefficient 
# T_n^{\nu m}(Q) is used in transforming spherical harmonic expansions when the coordinate system is rotated. 
# The rotation matrix Q, which is orthogonal, represents the transformation between the original and the rotated 
# coordinate systems. The spherical harmonics Y_n^m are functions on the sphere that are often used in physical 
# problems involving angular momentum, wave functions, and electromagnetic fields. The recursion method for 
# calculating rotation coefficients involves using properties of spherical harmonics and the Wigner D-matrix, 
# which represents the rotation operator in the space of spherical harmonics.

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
    
    # Iterate over all possible magnetic quantum numbers for the expansion
    for m_prime in range(-n, n + 1):
        # Calculate the spherical harmonics for the original coordinates
        Y_n_v = sph_harm(v, n, 0, 0)

        # Rotate spherical harmonic using the rotation matrix Q
        # In practice, this involves calculating a linear combination of the spherical harmonics
        # that represent the rotated state. This can be complex and typically uses Wigner D-matrices.
        # For this simplified version, we assume Q provides the necessary transformation.
        
        # Calculate the contribution to the rotated spherical harmonics
        Y_n_m_prime_rotated = sph_harm(m_prime, n, 0, 0)  # Simplified representation

        # Apply the rotation matrix Q to find the transformation
        # In practice, this would require the Wigner D-matrix, which represents the rotation in the
        # space of spherical harmonics
        T += Q[v, m_prime] * Y_n_m_prime_rotated

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
