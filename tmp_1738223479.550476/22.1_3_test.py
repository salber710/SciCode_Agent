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



    # Initialize a numpy array to store the translation coefficients
    R_lnm = np.zeros((N_t, N_t), dtype=complex)

    # Calculate r0 which is the absolute translation distance
    r0 = abs(z)

    # Precompute spherical Bessel and Neumann functions for use in recursion
    jn_values = [spherical_jn(l_idx, k * r0) for l_idx in range(N_t)]
    yn_values = [spherical_yn(l_idx, k * r0) for l_idx in range(N_t)]

    # Define a combined function of spherical Bessel and Neumann
    def bessel_combined(l_idx):
        return jn_values[l_idx] + 1j * yn_values[l_idx]

    # Compute translation coefficient using a different recursive approach
    for l_idx in range(N_t):
        for n_idx in range(N_t):
            if l_idx == 0 and n_idx == 0:
                # Set the base case for recursion
                R_lnm[l_idx, n_idx] = 1.0 + 0j
            else:
                # Use a different recursion formula, using precomputed Bessel functions
                term_l = bessel_combined(l_idx) * R_lnm[l_idx - 1, n_idx] if l_idx > 0 else 0
                term_n = bessel_combined(n_idx) * R_lnm[l_idx, n_idx - 1] if n_idx > 0 else 0

                # Combine terms with a novel weighting
                R_lnm[l_idx, n_idx] = (term_l + term_n) / (1 + l_idx + n_idx)

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