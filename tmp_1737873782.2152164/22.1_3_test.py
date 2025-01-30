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

    # Initialize the translation coefficient to zero
    R_lnm = 0.0

    # Calculate the radial distance r_0 = |r_p - r_q| = z in this case
    r_0 = z

    # Loop over l' from 0 to N_t
    for lp in range(N_t):
        for s in range(-lp, lp + 1):
            # Calculate the spherical harmonics Y_lp^s at the given angles
            Y_lp_s = sph_harm(s, lp, 0, 0)
            
            # Calculate the contribution of this term to the translation coefficient
            term = ((2 * lp + 1) * 1j**(lp - l)) * Y_lp_s * (sph_harm(m, l, 0, 0).conjugate()) / (4 * np.pi)
            
            # Sum over all contributions
            R_lnm += term

    # Multiply by the translation distance factor
    R_lnm *= np.exp(1j * k * r_0)

    return R_lnm


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