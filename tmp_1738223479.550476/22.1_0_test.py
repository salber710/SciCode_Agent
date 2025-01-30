from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy



# Background: In optical physics, translation coefficients are used to describe how a beam of light,
# such as a Gaussian beam, is affected when it is shifted or translated in space. The translation
# coefficient (R|R)_{ln}^{m} is a part of the mathematical formalism used to express the translated
# beam in terms of spherical harmonics, which are functions that arise in the solution of the
# Laplace equation in spherical coordinates. The recursion method for calculating these coefficients
# involves using previous values to compute the next, based on relationships derived from the
# properties of the spherical harmonics and the translation of the beam.

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
    R_lnm = np.zeros((N_t, N_t), dtype=complex)

    # Calculate the r0 parameter
    r0 = np.abs(z)

    # Loop over l and n, using a recursion relation
    for l_idx in range(N_t):
        for n_idx in range(N_t):
            if l_idx == 0 and n_idx == 0:
                # Base case for recursion, typically a known value
                R_lnm[l_idx, n_idx] = 1.0
            elif l_idx > 0:
                # Example recursion relation; this depends on the specific form of the problem.
                # This is a placeholder and should be replaced with the correct recursion formula.
                R_lnm[l_idx, n_idx] = (2*l_idx - 1) / l_idx * (k * r0) * R_lnm[l_idx-1, n_idx]
            elif n_idx > 0:
                # Handle cases where only n is increasing
                R_lnm[l_idx, n_idx] = (2*n_idx - 1) / n_idx * (k * r0) * R_lnm[l_idx, n_idx-1]

    # Return the computed translation coefficient for the specific (l, n, m)
    return R_lnm[l, n]


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