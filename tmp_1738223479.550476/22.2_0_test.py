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
        The translation distance along z direction.
    N_t : int
        Truncated space size.
    Output
    (R|R)lmn : complex
        The translation coefficient (R|R)lmn.
    '''



    # Calculate r0 as the Euclidean norm of translation along z
    r0 = np.sqrt(z**2)

    # Initialize a cache to store computed coefficients
    R_cache = {}

    # Precompute spherical Bessel functions
    jn_values = [spherical_jn(l_idx, k * r0) for l_idx in range(N_t)]

    # Function to compute translation coefficient recursively
    def compute_R(l_idx, n_idx):
        if l_idx == 0 and n_idx == 0:
            return 1.0 + 0j

        if (l_idx, n_idx) in R_cache:
            return R_cache[(l_idx, n_idx)]

        if l_idx == 0:
            value = jn_values[n_idx] * compute_R(l_idx, n_idx - 1) / (n_idx + 1)
        elif n_idx == 0:
            value = jn_values[l_idx] * compute_R(l_idx - 1, n_idx) / (l_idx + 1)
        else:
            term_a = jn_values[l_idx] * compute_R(l_idx - 1, n_idx) / (l_idx + 1)
            term_b = jn_values[n_idx] * compute_R(l_idx, n_idx - 1) / (n_idx + 1)
            value = (term_a + term_b) / 4  # Using a different division factor

        R_cache[(l_idx, n_idx)] = value
        return value

    # Compute the required translation coefficient
    return compute_R(l, n)



# Background: 
# In the context of spherical harmonics, the rotation of a coordinate system can be described using a rotation matrix Q. 
# The spherical harmonics Y_n^m, which are functions on the sphere, transform under rotations according to Wigner D-matrix elements.
# The rotation coefficients T_n^{\nu m} express how a spherical harmonic Y_n^\nu in the rotated system relates to Y_n^m in the original system.
# These coefficients can be calculated using recursion relations derived from the properties of Wigner D-matrices, which are fundamental in quantum mechanics 
# and angular momentum theory. The rotation matrix Q is a 3x3 orthogonal matrix that rotates the coordinate axes.

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



    # Initialize the rotation coefficient T
    T = 0.0 + 0.0j

    # Calculate the Euler angles from the rotation matrix Q
    # This step assumes Q is a valid rotation matrix
    beta = np.arccos(Q[2, 2])
    if np.sin(beta) == 0:
        alpha = 0
        gamma = np.arctan2(Q[0, 1], Q[0, 0])
    else:
        alpha = np.arctan2(Q[1, 2], Q[0, 2])
        gamma = np.arctan2(Q[2, 1], -Q[2, 0])

    # Precompute spherical harmonics for given m and v
    Y_m = sph_harm(m, n, 0, 0)
    Y_v = sph_harm(v, n, 0, 0)

    # Calculate the Wigner D-matrix element
    # Here we use a basic approximation; in practice, a more sophisticated approach is required
    # This could be replaced with a call to a library function that computes Wigner D-matrices
    D = np.exp(-1j * m * alpha) * np.exp(-1j * v * gamma) * np.cos(beta / 2)**(m + v)

    # Calculate the rotation coefficient T_n^{\nu m}
    T = D * Y_v.conjugate() * Y_m

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